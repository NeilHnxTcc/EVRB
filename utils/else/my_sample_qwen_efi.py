import copy
import inspect
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.nn import functional as F

from transformers.generation.candidate_generator import AssistantVocabTranslatorCache

import transformers

from transformers.cache_utils import (
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    HybridChunkedCache,
    OffloadedCache,
    OffloadedHybridCache,
    QuantizedCacheConfig,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers.pytorch_utils import isin_mps_friendly
from transformers.tokenization_utils import ExtensionsTrie
from transformers.utils import (
    ModelOutput,
    is_accelerate_available,
    is_hqq_available,
    is_optimum_quanto_available,
    is_torchdynamo_exporting,
    logging,
)
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.candidate_generator import (
    AssistedCandidateGenerator,
    AssistedCandidateGeneratorDifferentTokenizers,
    CandidateGenerator,
    EarlyExitCandidateGenerator,
    PromptLookupCandidateGenerator,
    UniversalSpeculativeDecodingGenerator,
    _crop_past_key_values,
    _prepare_attention_mask,
    _prepare_token_type_ids,
)


from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    MinPLogitsWarper,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,

)
from transformers.generation.stopping_criteria import (
    ConfidenceCriteria,
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)
from transformers.generation.configuration_utils import (
    NEED_SETUP_CACHE_CLASSES_MAPPING,
    QUANT_BACKEND_CLASSES_MAPPING,
    GenerationConfig,
    GenerationMode,
)



@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
    """

    sequences: torch.LongTensor
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


@dataclass
class GenerateEncoderDecoderOutput(ModelOutput):
    """
    Outputs of encoder-decoder generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
    """

    sequences: torch.LongTensor
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


@dataclass
class GenerateBeamDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
            of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
            Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True`):
            Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
    """

    sequences: torch.LongTensor
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


@dataclass
class GenerateBeamEncoderDecoderOutput(ModelOutput):
    """
    Outputs of encoder-decoder generation models, when using beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
            of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
            Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True`):
            Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, num_heads, generated_length,
            sequence_length)`.
        cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
    """

    sequences: torch.LongTensor
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None



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


from hyper_config import hyper_param


def prepare_inputs_for_generation(
    self,
    input_ids: torch.LongTensor,
    past_key_values: Optional[Cache] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """
    Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
    slicing inputs given the existing cache.

    See the forward pass in the model documentation for expected arguments (different models might have different
    requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
    """

    # 1. Handle BC:
    model_inputs = {}
    # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
    if self._supports_cache_class:
        model_inputs["cache_position"] = cache_position
    # - `cache_position` was not a mandatory input in `prepare_inputs_for_generation` for those models, and this
    #   function may be called outside of `generate`. Handle most use cases by creating `cache_position` on the fly
    #   (this alternative is not as robust as calling `generate` and letting it create `cache_position`)
    elif cache_position is None:
        past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        cache_position = torch.arange(past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device)


    #-----------------------------------------------hnx-st------------------------------------------------
    origin_input_ids = input_ids.clone()
    #-----------------------------------------------hnx-ed------------------------------------------------


    # 2. Generic cache-dependent input preparation
    if past_key_values is not None:
        model_inputs["past_key_values"] = past_key_values
        inputs_embeds, input_ids = self._cache_dependant_input_preparation(
            input_ids, inputs_embeds, cache_position
        )

    # 3. Prepare base model inputs
    input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step for every prompt.
    if not self.config.is_encoder_decoder:
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs[input_ids_key] = None
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            # `clone` calls in this function ensure a consistent stride. See #32227
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
            model_inputs["inputs_embeds"] = None
    else:
        model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

    # 4. Create missing `position_ids` on the fly
    encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
    attention_mask = (
        kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
    )
    attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
    position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
    if (
        attention_mask is not None
        and kwargs.get(position_ids_key) is None
        and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
    ):
        #-----------------------------------------------hnx-st------------------------------------------------
        ### origin code
        # position_ids = attention_mask.long().cumsum(-1) - 1
        # position_ids.masked_fill_(attention_mask == 0, 1)
        assert origin_input_ids.shape[0] == 1
        position_ids = torch.arange(origin_input_ids.shape[-1]).unsqueeze(0).to(attention_mask)

        #-----------------------------------------------hnx-ed------------------------------------------------
        kwargs[position_ids_key] = position_ids  # placed in kwargs for further processing (see below)

    # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
    for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
        model_input = kwargs.get(model_input_name)
        if model_input is not None:
            if past_key_values is not None:
                current_input_length = (
                    model_inputs["inputs_embeds"].shape[1]
                    if model_inputs.get("inputs_embeds") is not None
                    else model_inputs[input_ids_key].shape[1]
                )
                model_input = model_input[:, -current_input_length:]
                model_input = model_input.clone(memory_format=torch.contiguous_format)
            model_inputs[model_input_name] = model_input

    # 6. Create 4D attention mask is we are using a compilable cache (important for performant compiled forward
    # pass)
    if (
        isinstance(past_key_values, Cache)
        and past_key_values.is_compileable
        and attention_mask is not None
        and attention_mask.ndim == 2
    ):
        if model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
        else:
            batch_size, sequence_length = model_inputs[input_ids_key].shape[:2]

        # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
        # the 4D causal mask exists, it should be present in the base model (XXXModel class) or in its decoder.
        base_model = getattr(self, self.base_model_prefix, self)
        decoder = base_model.get_decoder() if hasattr(base_model, "get_decoder") else None
        causal_mask_creation_function = getattr(
            base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
        )
        if causal_mask_creation_function is None and decoder is not None:  # it may be in the decoder
            causal_mask_creation_function = getattr(
                decoder, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
        if causal_mask_creation_function is None:  # can't be found
            logger.warning_once(
                f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                "writing code, see Llama for an example implementation. If you're a user, please report this "
                "issue on GitHub."
            )
        else:
            attention_mask = causal_mask_creation_function(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.dtype,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )
    if attention_mask is not None:
        model_inputs[attention_mask_key] = attention_mask

    if encoder_attention_mask is not None:
        model_inputs["attention_mask"] = encoder_attention_mask

    # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
    for key, value in kwargs.items():
        if key not in model_inputs:
            model_inputs[key] = value

    # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
    model_inputs.pop("labels", None)
    return model_inputs





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
    # import pdb; pdb.set_trace()
    img_ent_thr = hyper_param.img_ent_thr
    pri_rec_thr = hyper_param.pri_rec_thr
    do_ct = hyper_param.do_ct
    do_eos = hyper_param.do_eos
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
        #  

        # get the last hidden state 
        last_hidden_state = outputs['hidden_states'][-1]
        # locate the visual tokens:  151655 in input ids
        img_idx = torch.where(input_ids == 151655) # tuple(2) img_num   
        # forward the tokens and get the visual tokens last hidden state
        img_last_hidden_state = last_hidden_state[img_idx] #[img_num, dim=3584]
        # self.lm_head process the visual tokens and get the probability 
        img_logits = self.lm_head(img_last_hidden_state) #
        # calculate the entropy: vocabulary size: 152064, max entropy 17.9321 (log2)
        image_prob = torch.nn.functional.softmax(img_logits, dim= -1)
        image_entropy = (- image_prob * torch.log2(image_prob+torch.finfo(image_prob.dtype).eps)).sum(dim = -1) #  num_token
        # statistic the image token number of different interval (20)
        # img_entropy_rounded = image_entropy.round()
        # value, count = torch.unique(img_entropy_rounded.int(), return_counts = True)
        # result = {'value':value, 'count':count}
        # return result
        

        
        # locate the large entropy tokens
        img_st, img_ed = img_idx[1][0], img_idx[1][-1] + 1
        model_kwargs['cache_position'] = model_kwargs['cache_position'][img_ed:]
        redundant_img_loc = torch.where(image_entropy > img_ent_thr)[0] + img_st
        clear_img_loc = torch.where(image_entropy <= img_ent_thr)[0] + img_st


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
        for idx in range(len(ct_model_kwargs['past_key_values'].key_cache)):
            ct_model_kwargs['past_key_values'].key_cache[idx] = ct_model_kwargs['past_key_values'].key_cache[idx][:,:,clear_mask]
        for idx in range(len(ct_model_kwargs['past_key_values'].value_cache)):
            ct_model_kwargs['past_key_values'].value_cache[idx] = ct_model_kwargs['past_key_values'].value_cache[idx][:,:,clear_mask]

        # modified the input ids : chunk after the image ids
        # input_ids = input_ids[:, img_ed:]
        # modified the past key value of model_kwargs, chunk before image end

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
        # import pdb; pdb.set_trace()
        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})


        #------------------------------hnx-st-------------------------------------
        if do_ct:
            # prepare model inputs
            ct_model_inputs = self.prepare_inputs_for_generation(input_ids, **ct_model_kwargs)
            # import pdb; pdb.set_trace()
            # prepare variable output controls (note: some models won't accept all output controls)
            # ct_model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            # ct_model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})      
        #------------------------------hnx-ed-------------------------------------


        #------------------------------hnx-st-------------------------------------
        #------------------------------hnx-ed-------------------------------------




        if is_prefill:
            if do_eos:
                model_inputs['logits_to_keep'] = int(model_inputs['input_ids'].shape[-1])
            outputs = self(**model_inputs, return_dict=True, output_attentions = do_eos )
            #------------------------------hnx-st-------------------------------------
            if do_ct:
                ct_outputs = self(**ct_model_inputs, return_dict=True)
            #------------------------------hnx-ed-------------------------------------
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True, output_attentions = do_eos)
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
            # import pdb; pdb.set_trace()
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
        if do_eos and input_ids[:,-1] == 13:
            next_token_logits[:,2] = next_token_logits[:,2] * scale_factor
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
    transformers.generation.GenerationMixin.prepare_inputs_for_generation = prepare_inputs_for_generation
    transformers.generation.utils.GenerationMixin._sample = _sample


