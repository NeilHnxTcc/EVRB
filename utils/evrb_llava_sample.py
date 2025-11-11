import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput


from utils.hyper_config import hyper_param



def prepare_inputs_for_generation(
    input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values:
        input_ids = input_ids[:, -1:]
    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}
    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "images": kwargs.get("images", None),
        }
    )
    return model_inputs


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)



# Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 2:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        ) 

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        if attention_mask.shape[0] > 1:
            input_shape = (input_shape[0], input_shape[1]//2)
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        if attention_mask.shape[0] > 1:
            expanded_attn_mask = torch.cat((expanded_attn_mask[:1], expanded_attn_mask[1:]),dim=-2)

        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )
    return combined_attention_mask




class AnomalyDetector:
    def __init__(self, vv_threshold = None):
        self.sequence = None     # 原始序列
        self.deltas = None       # 差值序列
        self.anomalies = torch.tensor([])    # 异常点索引
        self.alpha = 8 # 越大越苛刻
        self.wait_num = 13 # 问题之后再找离群点
        if vv_threshold == None:
            self.vv_threshold = 0.05
        else:
            self.vv_threshold = vv_threshold
    def add_point(self, new_value):
        """
        添加新数据点并检测异常
        """
        # 添加新值到序列
        new_value = torch.tensor([new_value])
        if self.sequence == None:
            self.sequence = new_value
        else:
            self.sequence = torch.cat((self.sequence, new_value))


        if self.deltas == None:
            self.deltas = torch.tensor([0])
        else:
            delta = self.sequence[-1] - self.sequence[-2]
            delta = torch.tensor([delta])
            self.deltas = torch.cat((self.deltas, delta))
        
        if len(self.deltas) <= self.wait_num:
            return 
        # 计算当前中位数
        median = torch.median(torch.abs(self.deltas))
        
        # 检测异常条件
        # if abs(delta) > self.alpha * median:
        if abs(delta) > self.vv_threshold:   
            # 判断异常点位置
            if delta > 0:
                # 正跳变：前一个点异常
                anomaly_idx = len(self.sequence) - 2
            else:
                # 负跳变：当前点异常
                anomaly_idx = len(self.sequence) - 1
            

            if anomaly_idx not in self.anomalies:
                self.anomalies = torch.cat((self.anomalies, torch.tensor([anomaly_idx])))

def local_prepare_inputs_for_generation(
     input_ids,  past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    bs, seq_len = input_ids.shape
    seq_len = seq_len + 24*24 -1 

    sub_mask = torch.tensor([[1,0],[0,1]]).to(attention_mask)
    attention_mask = torch.cat((attention_mask, sub_mask), dim=-1)


    position_ids = kwargs['position_ids'][-1:] + 1 
    position_ids = position_ids.repeat(2)

    input_ids = input_ids[:, -1:].repeat(1,2)

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}
    
    model_inputs['position_ids'] = position_ids
    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "images":  None,
        }
    )
    return model_inputs


def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.sample`] directly. Use generate() instead.
    For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        logits_warper (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
            to warp the prediction score distribution of the language modeling head applied before multinomial
            sampling at each generation step.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.SampleDecoderOnlyOutput`], [`~generation.SampleEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.SampleEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForCausalLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     TopKLogitsWarper,
    ...     TemperatureLogitsWarper,
    ...     StoppingCriteriaList,
    ...     MaxLengthCriteria,
    ... )
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

    >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
    >>> model.config.pad_token_id = model.config.eos_token_id
    >>> model.generation_config.pad_token_id = model.config.eos_token_id

    >>> input_prompt = "Today is a beautiful day, and"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
    ...     ]
    ... )
    >>> # instantiate logits processors
    >>> logits_warper = LogitsProcessorList(
    ...     [
    ...         TopKLogitsWarper(50),
    ...         TemperatureLogitsWarper(0.7),
    ...     ]
    ... )

    >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
    >>> outputs = model.sample(
    ...     input_ids,
    ...     logits_processor=logits_processor,
    ...     logits_warper=logits_warper,
    ...     stopping_criteria=stopping_criteria,
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Today is a beautiful day, and we must do everything possible to make it a day of celebration.']
    ```"""

    # init hyper parameters
    img_ent_thr = hyper_param.img_ent_thr
    pri_rec_thr = hyper_param.pri_rec_thr
    do_ct = hyper_param.do_ct
    do_eos = hyper_param.do_eos
    
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only
    # auto-regressive generation



    img_st = torch.where(input_ids[0] == -200)[0].item()
    img_end = img_st +  24*24



    # create the weight map
    bs = input_ids.shape[0]
    assert bs == 1


    model_inputs = prepare_inputs_for_generation(input_ids, **model_kwargs) # dict_keys(['input_ids', 'past_key_values', 'use_cache', 'attention_mask', 'images'])

    # from time import time
    # st_time = time()
    outputs = self(
        **model_inputs,
        return_dict=True,
        output_hidden_states=True,
    )
    # print(f'duration: {time() - st_time}')
    # import pdb; pdb.set_trace()

    hidden_states = outputs.hidden_states[-1]  # 33, num_token, 4096
    image_hidden_states = hidden_states[:,img_st:img_end]
    image_prob = torch.nn.functional.softmax(self.lm_head(image_hidden_states), dim= -1)
    image_entropy = (- image_prob * torch.log2(image_prob+torch.finfo(image_prob.dtype).eps)).sum(dim = -1) #  576
    
    batch_size, seq_length_with_past = input_ids.shape
    seq_length_with_past += 24*24 - 1 
    selected_entropy  = image_entropy[-1]

    top_attention_rank_index = torch.where(selected_entropy > img_ent_thr)[0] + img_st
    bottem_attention_rank_index = torch.where(selected_entropy <= img_ent_thr)[0] + img_st


    # #------------------------------hnx-st-------------------------------------
    # # locate the text tokens : img idx = 151655
    # bs, token_num = input_ids.shape
    # assert bs == 1

    # img_st = torch.where(input_ids[0] == -200)[0].item()
    # img_end = img_st +  24*24 -1 
    # full_text_st  = img_end + 1
    # text_st = img_st + 1

    # # duplicate the text tokens
    # new_input_ids = torch.cat([input_ids, input_ids[:, text_st:]], dim = -1)
    
    # # customize the attention_mask
    # full_len = token_num  + 24*24 -1 
    # attention_mask = torch.ones((bs, full_len)).to(input_ids)
    # attention_mask = torch.cat([attention_mask, attention_mask[:,full_text_st:].clone()], dim = -1)
    # additional_attn_mask = attention_mask.clone()

    # attention_mask[:, full_len:] = 0
    # attention_mask[:, top_attention_rank_index] = 0
    # additional_attn_mask[:, full_text_st: full_len] = 0
    # additional_attn_mask[:, bottem_attention_rank_index] = 0
    # attention_mask = torch.cat([attention_mask, additional_attn_mask],dim = 0)
    # # update the model kwargs and input ids
    # model_kwargs['attention_mask'] = attention_mask
    # input_ids = new_input_ids
    # #------------------------------hnx-ed-------------------------------------

    def initial_my_input(inputs, outputs, top_attention_rank_index, bottem_attention_rank_index, img_st_idx=34, img_len=24*24):
        input_ids = inputs['input_ids']
        bs, seq_len = input_ids.shape
        full_len = seq_len + img_len -1 
        full_text_st  = img_st_idx + img_len
        attention_mask = torch.ones((bs, full_len)).to(input_ids)
        attention_mask = torch.cat([attention_mask, attention_mask[:,full_text_st:].clone()], dim = -1)
        additional_attn_mask = attention_mask.clone()

        attention_mask[:, full_len:] = 0
        attention_mask[:, top_attention_rank_index] = 0
        additional_attn_mask[:, full_text_st: full_len] = 0
        additional_attn_mask[:, bottem_attention_rank_index] = 0
        attention_mask = torch.cat([attention_mask, additional_attn_mask],dim = 0)


        past_key_values = torch.stack([ torch.stack(layer,dim=0) for  layer in outputs['past_key_values'] ], dim = 0) # [32, 2, 1,32,num_token,dim]))
        result = {}
        ids_after_img = input_ids[:,img_st_idx+1:].repeat(1,2)
        img_end = img_st_idx + img_len
        ct_key_values = past_key_values[:,:,:,:,:img_end]
        result['input_ids'] = ids_after_img
        result['past_key_values'] = ct_key_values
        result['use_cache'] = True
        result['attention_mask'] = attention_mask
        result['images'] = None
        position_ids = torch.arange(full_text_st, full_len).to(attention_mask)
        result['position_ids'] = position_ids.repeat(2)
        return result
    
    model_inputs = initial_my_input(model_inputs, outputs, top_attention_rank_index, bottem_attention_rank_index)


    flag = False
    if do_eos:
        def kl_divergence(p, q):
            p = torch.clip(p, 1e-6, 1.0)
            q = torch.clip(q, 1e-6, 1.0)
            return torch.sum(p * torch.log2(p / q), dim=-1)

        def js_divergence(p, q):
            m = 0.5 * (p + q)
            return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))  
        vv_thr = hyper_param.vv_thr
        eos_k = hyper_param.eos_k
        count = 0
        detector = AnomalyDetector(vv_threshold=vv_thr)
        all_js_div = None
        object_dict = {}
        idx_list = []
        id_shift = input_ids.shape[-1] - model_inputs['input_ids'].shape[-1] 


    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break


        if flag:
            model_kwargs['attention_mask'] = model_inputs['attention_mask']
            model_kwargs['position_ids'] = model_inputs['position_ids']
            model_inputs = local_prepare_inputs_for_generation(input_ids,  **model_kwargs)
        
        flag = True

        # st_time = time()
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions = do_eos,
        )
        # print(f'duration: {time() - st_time}')
        # import pdb; pdb.set_trace()

        seq_num = outputs['logits'].shape[-2]
        if  seq_num > 2:
            origin_loc = seq_num // 2 -1 
            logits = outputs['logits'][:, origin_loc]
            obs_logits = outputs['logits'][:, -1]
        else:
            logits = outputs['logits'][:, -2]
            obs_logits = outputs['logits'][:, -1]

        scale_factor = 0
        count = 0
        
        if do_eos:
            clear_text_prob = torch.nn.functional.softmax(logits, dim= -1)  # token, 32000
            blur_text_prob = torch.nn.functional.softmax(obs_logits, dim= -1)  # token, 32000
            js_div = js_divergence(clear_text_prob, blur_text_prob) # token
            if all_js_div == None:
                all_js_div = js_div[0]
            else:
                all_js_div = torch.cat((all_js_div, js_div[0]))
            ## get the entropy of text tokens  (only consider the new generated tokens)
            attention_maps = torch.cat(outputs.attentions, dim = 0).mean(dim= 1).mean(dim = 0) #token_num, 24*24
            attention_maps = torch.nn.functional.softmax(attention_maps,dim = -1)
            attn_entropy = (- attention_maps * torch.log2(attention_maps + 1e-6)).sum(dim=-1) # token_num
            before_len = len(detector.anomalies)
            for entropy in attn_entropy:
                detector.add_point(entropy)
                
            
            cur_len = len(detector.anomalies)
            if before_len < cur_len :         
                for i in range(cur_len-before_len):
                    ii = cur_len-before_len - (i + 1)  # make sure the order is the same
                    end_idx = int(detector.anomalies[-1-ii])
                    
                    real_idx = end_idx + id_shift  
                    
                    real_id = int(input_ids[0,real_idx])
                    
                    before_idx = real_idx - 1
                    before_id = int(input_ids[0,before_idx])

                    if (len(idx_list) == 0 or before_idx != idx_list[-1]) and all_js_div[end_idx-2] > all_js_div[end_idx-1]:
                        real_id = before_id
                        real_idx = before_idx
                        real_js = all_js_div[end_idx-2]
                    else:
                        real_js = all_js_div[end_idx-1]
                    
                    idx_list.append(real_idx)

                    if real_id not in object_dict.keys():
                        object_dict[real_id] = [real_js] # design for the vv js not aligned problem
                    else:
                        object_dict[real_id].append(real_js)
            all_values = []
            for key, value in object_dict.items():
                all_values = all_values + value
                if len(value) > 1:
                    scale_factor += max(value[0]-value[-1],0)
                    count += 1 


        ##### same token different js div 
        if count == 0:
            scale_factor = torch.tensor(1)
        else:
            scale_factor = eos_k * torch.tensor(scale_factor / count) + 1

    

            
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need
        
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
        ##### adjust the eos logit by objects' js div
        if do_eos and input_ids[:,-1] == 29889:
            next_token_logits[:,2] = next_token_logits[:,2] * scale_factor
    
        if do_ct:
            logits =  logits.to(copy=True, dtype=torch.float32, device=input_ids.device)
            ct_logits = obs_logits.to(copy=True, dtype=torch.float32, device=input_ids.device)
            logits = logits_processor(input_ids, logits)
            logits = logits_warper(input_ids, logits)
            
            ct_logits = logits_processor(input_ids, ct_logits)
            ct_logits = logits_warper(input_ids, ct_logits)  
                    
            probs = nn.functional.softmax(logits, dim=-1)
            
            ct_probs = nn.functional.softmax(ct_logits, dim = -1)

            threshold =  probs.max() * pri_rec_thr
            selected_indices = torch.where(probs > threshold) 
            

            
            origin_value = probs[selected_indices]
            ct_value = ct_probs[selected_indices]
            origin_sum = origin_value.sum(dim = -1,keepdim=True)
            next_token_probs = probs.clone()
            modified_value = origin_value / (ct_value + 10e-5)
            modified_sum = modified_value.sum(dim=-1, keepdim =True)
            next_token_probs[selected_indices] = modified_value * origin_sum / (modified_sum + 10e-5)

        else:
            next_token_logits = logits_processor(input_ids, next_token_logits)
            next_token_logits = logits_warper(input_ids, next_token_logits)
            next_token_probs = nn.functional.softmax(next_token_logits, dim=-1)


        if model_kwargs['sample_greedy']:
            next_tokens = torch.argmax(next_token_probs, dim=-1)
        else:
            next_tokens = torch.multinomial(next_token_probs, num_samples=1).squeeze(1)
            

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)


        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        ) # mainly update the past key values 32(2(1,32,token_num,128))
        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()
    
    
    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids
    

def evolve_my_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
