import copy
import inspect
import os
from typing import  Optional,  Union



import torch
from torch import nn

import transformers

from transformers.cache_utils import Cache


from transformers.generation.logits_process import LogitsProcessorList



from transformers.generation.stopping_criteria import  StoppingCriteriaList


from transformers.generation.configuration_utils import  GenerationConfig


from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput




GreedySearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
ContrastiveSearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput

ContrastiveSearchEncoderDecoderOutput = GenerateEncoderDecoderOutput
GreedySearchEncoderDecoderOutput = GenerateEncoderDecoderOutput
SampleEncoderDecoderOutput = GenerateEncoderDecoderOutput

BeamSearchDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput
BeamSampleDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput

BeamSearchEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput
BeamSampleEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput

GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]
ContrastiveSearchOutput = Union[ContrastiveSearchEncoderDecoderOutput, ContrastiveSearchDecoderOnlyOutput]

# Typing shortcuts
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]
GenerateOutput = Union[GenerateNonBeamOutput, GenerateBeamOutput]


from utils.hyper_config import hyper_param



class AnomalyDetector:
    def __init__(self, vv_threshold = None):
        self.sequence = torch.empty(1)     # 原始序列
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
        self.sequence = torch.cat((self.sequence, new_value))

        if self.deltas == None:
            self.deltas = torch.tensor([0])
        else:
            delta = self.sequence[-1] - self.sequence[-2]
            delta = torch.tensor([delta])
            self.deltas = torch.cat((self.deltas, delta))
        
        if len(self.deltas) <= self.wait_num:
            return 
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





def _sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    #------------------------------hnx-st-------------------------------------
    # init hyper parameters
    img_ent_thr = hyper_param.img_ent_thr
    pri_rec_thr = hyper_param.pri_rec_thr
    do_ct = hyper_param.do_ct
    do_eos = hyper_param.do_eos
    stop_id = hyper_param.stop_id
    img_id = hyper_param.img_id
    assert stop_id != None , img_id != None
    #------------------------------hnx-ed------------------------------------- 
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
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
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    model_forward = self.__call__
    compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True



    #------------------------------hnx-st-------------------------------------
    if do_ct:
        # model_kwargs:  
        # attention_mask: [bs, num_token]
        # past_key_values : transformers.cache_utils.DynamicCache object 
        # cache_position : [num_token] 
        #  
        # pixel_values [H,W] original size
        # image grid thw : [bs, x, x ]
        # use_cache : True

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = self(**model_inputs, return_dict=True,output_hidden_states=True)
        #  logits : [bs, num_token, voc=152064]
        #  past_key_values : transformers.cache_utils.DynamicCache   .key_cache || .value_cache : list(layer_num=28)[bs, hd=4,num_token,dim=128]
        #  hidden_states : tuple(29)[bs, num_token, dim=3584]
        

        # get the last hidden state 
        last_hidden_state = outputs['hidden_states'][-1]
        # locate the visual tokens:  32000 in input ids
        img_idx = torch.where(input_ids == img_id) # tuple(2) img_num   
        # forward the tokens and get the visual tokens last hidden state
        img_last_hidden_state = last_hidden_state[img_idx] #[img_num, dim=3584]
        # self.lm_head process the visual tokens and get the probability 
        img_logits = self.lm_head(img_last_hidden_state) 
        # calculate the entropy: vocabulary size: 32064, max entropy 14.9687 (log2)
        image_prob = torch.nn.functional.softmax(img_logits, dim= -1)
        image_entropy = (- image_prob * torch.log2(image_prob+torch.finfo(image_prob.dtype).eps)).sum(dim = -1) #  num_token
        # statistic the image token number of different interval (20)
        # img_entropy_rounded = image_entropy.round()
        # value, count = torch.unique(img_entropy_rounded.int(), return_counts = True)
        # result = {'value':value, 'count':count}
        # return result
        
        
        # locate the large entropy tokens
        img_st, img_ed = img_idx[1][0], img_idx[1][-1] + 1
        # cache position of model_kwargs chunk  after image ids
        redundant_img_loc = torch.where(image_entropy > img_ent_thr)[0] + img_st
        clear_img_loc = torch.where(image_entropy <= img_ent_thr)[0] + img_st
        # modified the attention mask of model_kwargs: large entropy set to be 0
        origin_attention_mask = model_kwargs['attention_mask']
        redundant_mask = origin_attention_mask[0].bool().clone()
        redundant_mask[redundant_img_loc] = False
        redundant_mask[img_ed:] = False
        clear_mask = origin_attention_mask[0].bool().clone()
        clear_mask[clear_img_loc] = False
        clear_mask[img_ed:] = False
        if redundant_img_loc.shape[-1] > 0:
            model_kwargs['attention_mask'] = origin_attention_mask[:,:-redundant_img_loc.shape[-1]]
        else: 
            model_kwargs['attention_mask'] = origin_attention_mask
        # duplicate another ct_model_kwargs for ct decoding
        ct_model_kwargs = copy.deepcopy(model_kwargs)
        if clear_img_loc.shape[-1] > 0:
            ct_model_kwargs['attention_mask'] = origin_attention_mask[:,:-clear_img_loc.shape[-1]] 
        else:
            ct_model_kwargs['attention_mask'] = origin_attention_mask

        ct_model_kwargs['cache_position'] =  ct_model_kwargs['cache_position'][img_ed:]

        for idx in range(len(ct_model_kwargs['past_key_values'].key_cache)):
            ct_model_kwargs['past_key_values'].key_cache[idx] = ct_model_kwargs['past_key_values'].key_cache[idx][:,:,clear_mask]
        for idx in range(len(ct_model_kwargs['past_key_values'].value_cache)):
            ct_model_kwargs['past_key_values'].value_cache[idx] = ct_model_kwargs['past_key_values'].value_cache[idx][:,:,clear_mask]


        # modified the input ids : chunk after the image ids
        # input_ids = input_ids[:, img_ed:]
        # modified the past key value of model_kwargs, chunk before image end
        model_kwargs['cache_position'] =  model_kwargs['cache_position'][img_ed:]
        for idx in range(len(model_kwargs['past_key_values'].key_cache)):
            model_kwargs['past_key_values'].key_cache[idx] = model_kwargs['past_key_values'].key_cache[idx][:,:,redundant_mask]
        for idx in range(len(model_kwargs['past_key_values'].value_cache)):
            model_kwargs['past_key_values'].value_cache[idx] = model_kwargs['past_key_values'].value_cache[idx][:,:,redundant_mask]
        



    if do_eos:
        all_js_div = None
        idx_list = []
        object_dict = {}
        vv_thr = hyper_param.vv_thr
        eos_k = hyper_param.eos_k
        id_shift = img_ed
        detector = AnomalyDetector(vv_threshold = vv_thr)
    #------------------------------hnx-ed-------------------------------------
    


    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # prepare variable output controls (note: some models won't accept all output controls)
        # model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        # model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})


        #------------------------------hnx-st-------------------------------------
        if do_ct:
            # prepare model inputs
            ct_model_inputs = self.prepare_inputs_for_generation(input_ids, **ct_model_kwargs)
            # prepare variable output controls (note: some models won't accept all output controls)
            # ct_model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            # ct_model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})      
        #------------------------------hnx-ed-------------------------------------


        #------------------------------hnx-st-------------------------------------
        #------------------------------hnx-ed-------------------------------------



        if is_prefill:
            model_inputs['output_attentions'] = do_eos
            outputs = self(**model_inputs, return_dict=True)
            #------------------------------hnx-st-------------------------------------
            if do_ct:
                ct_outputs = self(**ct_model_inputs, return_dict=True)
            #------------------------------hnx-ed-------------------------------------
            is_prefill = False
        else:
            model_inputs['output_attentions'] = do_eos
            outputs = model_forward(**model_inputs, return_dict=True)
            #------------------------------hnx-st-------------------------------------
            if do_ct:
                ct_outputs = model_forward(**ct_model_inputs, return_dict=True)
            #------------------------------hnx-ed-------------------------------------          



        #------------------------------hnx-st-------------------------------------
        if do_eos:
            def kl_divergence(p, q):
                """计算KL散度 (以2为底)"""
                p = torch.clip(p, 1e-6, 1.0)
                q = torch.clip(q, 1e-6, 1.0)
                return torch.sum(p * torch.log2(p / q), dim=-1)

            def js_divergence(p, q):
                """计算JS散度"""
                m = 0.5 * (p + q)
                return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))  
            ##### get the js div of each token
            logits = outputs['logits']
            ct_logits = ct_outputs['logits']
            clear_text_prob = torch.nn.functional.softmax(logits, dim= -1)  # token, 32000
            blur_text_prob = torch.nn.functional.softmax(ct_logits, dim= -1)  # token, 32000
            js_div = js_divergence(clear_text_prob, blur_text_prob) # token
            if all_js_div == None:
                all_js_div = js_div[0]
            else:
                all_js_div = torch.cat((all_js_div, js_div[0]))
            
            ## get the entropy of text tokens  (only consider the new generated tokens)
            attention_maps = torch.cat(outputs['attentions'], dim = 0).mean(dim= 1).mean(dim = 0) #new_token_num, all_token_num
            attention_maps = attention_maps[:, img_st:img_ed]
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
                    
                    real_idx = end_idx + id_shift - 1

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

            scale_factor = 0
            count = 0
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
        #------------------------------hnx-ed-------------------------------------





        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        #------------------------------hnx-st-------------------------------------
        if do_ct:
            ct_model_kwargs = self._update_model_kwargs_for_generation(
                ct_outputs,
                ct_model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
        #------------------------------hnx-ed-------------------------------------

        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)




        #------------------------------hnx-st-------------------------------------
        # "." idx = 29889
        if do_eos and input_ids[:,-1] == stop_id:
            eos_id = stopping_criteria[1].eos_token_id[0]
            if next_token_logits[0,eos_id] > 0:
                next_token_logits[0,eos_id] = next_token_logits[0,eos_id] * scale_factor
        #------------------------------hnx-ed-------------------------------------
        


        #------------------------------hnx-st-------------------------------------
        if do_ct:
            ct_next_token_logits = ct_outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            
            logits =  next_token_logits   # bs, num_token, channel
            ct_logits = ct_next_token_logits
            logits = logits_processor(input_ids, logits)
            ct_logits = logits_processor(input_ids, ct_logits)      
            probs = nn.functional.softmax(logits, dim=-1)
            ct_probs = nn.functional.softmax(ct_logits, dim = -1)
            # 使用 threshold 矫正
            threshold =  probs.max() * pri_rec_thr
            
            selected_indices = torch.where(probs > threshold) 
            

            
            origin_value = probs[selected_indices]
            ct_value = ct_probs[selected_indices]
            origin_sum = origin_value.sum(dim = -1,keepdim=True)
            next_token_scores = probs.clone()
            # 使用 除法 计算
            modified_value = origin_value / (ct_value + 10e-5)
            modified_sum = modified_value.sum(dim=-1, keepdim =True)
            next_token_scores[selected_indices] = modified_value * origin_sum / (modified_sum + 10e-5)
        #------------------------------hnx-ed-------------------------------------
        else:
            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )
    # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def evolve_my_sampling():
    transformers.generation.utils.GenerationMixin._sample = _sample


