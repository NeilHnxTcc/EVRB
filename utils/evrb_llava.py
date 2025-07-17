
import torch
from torch.cuda.amp import autocast as autocast

from torch import nn
from typing import List, Optional, Tuple, Union

from utils.llava.llava_llama import LlavaLlamaForCausalLM
from utils.llava.base_model import BaseModel

from transformers import AutoTokenizer, BitsAndBytesConfig, LlamaConfig

import transformers

import math

# import pdb; pdb.set_trace()
# from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding, apply_rotary_pos_emb, repeat_kv

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding,  apply_rotary_pos_emb




# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
NUM_IMAGE_TOKENS = 576


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        #------------------------------hnx-st-------------------------------------
        if output_attentions:
            new_value_states = value_states.clone() # before reuse
        #------------------------------hnx-ed-------------------------------------

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        
        cos, sin = self.rotary_emb(value_states, seq_len=1000)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        #-------------------------------st-------------------------------------
        if output_attentions:
            vv_attn_weights = torch.matmul(new_value_states, value_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        else:
            vv_attn_weights = None
        #-------------------------------ed-------------------------------------

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, vv_attn_weights, past_key_value


def use_my_llava():
    transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttention
    

class LLaVa(BaseModel):
    """
    LLaVa-1.5 model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/llava-1.5_vicuna7b.yaml",
        "vicuna13b": "configs/models/llava-1.5_vicuna13b.yaml",
    }

    def __init__(
        self,
        vision_tower=r'openai/clip-vit-large-patch14',
        mm_vision_select_layer=-2,
        merged_ckpt="",
        cache_dir=None,
        model_max_length=2048,
        shikra_version="v1",
        freeze_backbone=False,
        mm_use_im_start_end=True,
        pretrain_mm_mlp_adapter=None,
        tune_mm_mlp_adapter=False,
        freeze_mm_mlp_adapter=False,
        apply_fsdp=None,
        max_txt_len=128,
        max_output_txt_len=256,
        low_resource=False,  # use 8 bit and put vit in cpu
        bf16=False, 
        fp16=True,
        system_message="",
        load_8bit=False, 
        load_4bit=False, 
        device_map="auto", 
        device="cuda",
    ):
        super().__init__()

        kwargs = {"device_map": device_map}
        self.system_message = system_message
        if load_8bit:
            kwargs['load_in_8bit'] = True
        elif load_4bit:
            kwargs['load_in_4bit'] = True
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        else:
            kwargs['torch_dtype'] = torch.float16
        self.llama_tokenizer = AutoTokenizer.from_pretrained(merged_ckpt, use_fast=False)
        self.llama_model = LlavaLlamaForCausalLM.from_pretrained(
            merged_ckpt, low_cpu_mem_usage=True, **kwargs)

        mm_use_im_start_end = getattr(self.llama_model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(self.llama_model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.llama_tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

        vision_tower = self.llama_model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, samples):
        image = samples["image"]

        instruction = samples["prompt"] if "prompt" in samples else None

        bs = image.size(0)

        if isinstance(instruction, str):
            instruction = [instruction] * bs
        else:
            assert len(instruction) == bs, "The number of prompts must be equal to the batch size."

        instruction = [p.replace('<ImageHere>', '<image>') for p in instruction]
        instruction = [self.system_message + p for p in instruction]

        input_ids = self.tokenizer_image_token(instruction, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        ###TODO: targets, attention_mask
        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_ids=inputs_ids,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self, 
        prompt,
        image,
        use_nucleus_sampling=False,
        num_beams=5,
        max_new_tokens=300,
        top_p=0.9,
        temperature=1,
        return_dict_in_generate=False,
        key_position=None,
        use_cache = True,
        **model_kwargs,
    ):
        self.llama_tokenizer.padding_side = "left"

        image = image
        instruction = prompt

        bs = image.size(0)

        if isinstance(instruction, str):
            instruction = [instruction] * bs
        else:
            assert len(instruction) == bs, "The number of prompts must be equal to the batch size."

        instruction = [self.system_message + p for p in instruction]

        chunks_before, chunks_after = [], []
        for p in instruction:
            chunk_before, chunk_after = p.split('<ImageHere>')
            chunks_before.append(chunk_before)
            chunks_after.append(chunk_after)

        tokens_before = self.llama_tokenizer(
            chunks_before,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False
        ).to(image.device).input_ids

        tokens_after = self.llama_tokenizer(
            chunks_after,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False
        ).to(image.device).input_ids

        bos = torch.ones([bs, 1],
                         dtype=torch.int64,
                         device=image.device) * self.llama_tokenizer.bos_token_id

        image_token = torch.ones([bs, 1],
                         dtype=torch.int64,
                         device=image.device) * IMAGE_TOKEN_INDEX

        with self.maybe_autocast():
            input_ids = torch.cat([bos, tokens_before, image_token, tokens_after], dim=1)

            if key_position is None:
                key_position = {
                    "image_start": tokens_before.shape[1]+1, 
                    "image_end": tokens_before.shape[1]+NUM_IMAGE_TOKENS, 
                    "response_start": input_ids.shape[1]+NUM_IMAGE_TOKENS-1,
                }
            
            output_ids = self.llama_model.generate(
                input_ids=input_ids,
                use_cache=True,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.llama_tokenizer.pad_token_id,
                bos_token_id=self.llama_tokenizer.bos_token_id,
                eos_token_id=self.llama_tokenizer.eos_token_id,
                images=image,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        output_text = self.llama_tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        output_text = [text.split('###')[0].strip() for text in output_text]

        return output_text 


    def embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds


    @classmethod
    def from_config(cls, cfg):
        vision_tower = cfg.get("vit_model", r'openai/clip-vit-large-patch14')
        mm_vision_select_layer = cfg.get("mm_vision_select_layer", -2)
        merged_ckpt = cfg.get("merged_ckpt", "")
        cache_dir = cfg.get("cache_dir", None)
        model_max_length = cfg.get("model_max_length", 2048)
        shikra_version = cfg.get("version", "v1")
        freeze_backbone = cfg.get("freeze_backbone", False)
        mm_use_im_start_end = cfg.get("mm_use_im_start_end", True)
        pretrain_mm_mlp_adapter = cfg.get("pretrain_mm_mlp_adapter", None)
        tune_mm_mlp_adapter = cfg.get("tune_mm_mlp_adapter", False)
        freeze_mm_mlp_adapter = cfg.get("freeze_mm_mlp_adapter", False)
        apply_fsdp = cfg.get("apply_fsdp", None)
        max_txt_len = cfg.get("max_txt_len", 128)
        max_output_txt_len = cfg.get("max_output_txt_len", 256)
        low_resource = cfg.get("low_resource", False)
        bf16 = cfg.get("bf16", False)
        fp16 = cfg.get("fp16", False)
        load_8bit = cfg.get("load_8bit", False)
        system_message = cfg.get("system_message", "")
        model = cls(
            vision_tower=vision_tower,
            mm_vision_select_layer=mm_vision_select_layer,
            merged_ckpt=merged_ckpt,
            cache_dir=cache_dir,
            model_max_length=model_max_length,
            shikra_version=shikra_version,
            freeze_backbone=freeze_backbone,
            mm_use_im_start_end=mm_use_im_start_end,
            pretrain_mm_mlp_adapter=pretrain_mm_mlp_adapter,
            tune_mm_mlp_adapter=tune_mm_mlp_adapter,
            freeze_mm_mlp_adapter=freeze_mm_mlp_adapter,
            apply_fsdp=apply_fsdp,
            max_txt_len=max_txt_len,
            max_output_txt_len=max_output_txt_len,
            low_resource=low_resource,  # use 8 bit and put vit in cpu
            bf16=bf16, fp16=fp16,
            system_message=system_message,
        )

        return model