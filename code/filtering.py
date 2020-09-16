import argparse
import torch
import torch.nn.functional as F
import numpy as np
import transformers 
import logging
import utils
import sampler

EOS_ID = 50256
logger = logging.getLogger("main")

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    if top_k!=None:
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            # Remove all tokens with a probability less than the last token in the top-k tokens
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]  # compute top-k 
            logits[indices_to_remove] = filter_value

    if top_p!=None and top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits

@torch.no_grad()
def generate(
    model,
    input_ids=None,
    max_length=None,
    do_sample=None,
    num_beams=None, # forcing to be None
    temperature=None, 
    schedule=None,
    repetition_penalty=None, # forcing to be 1.0
    bos_token_id=None,
    pad_token_id=None,
    eos_token_id=None,
    num_return_sequences=None,
    dry_run=None,
    attention_mask=None,
    decoder_start_token_id=None,
):
    r""" Generates sequences for models with a LM head. The method currently supports greedy decoding, beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.
    Adapted in part from `Facebook's XLM beam search code`_.
    .. _`Facebook's XLM beam search code`:
       https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529
    Parameters:
        input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
            The sequence used as a prompt for the generation. If `None` the method initializes
            it as an empty `torch.LongTensor` of shape `(1,)`.
        max_length: (`optional`) int
            The min length of the sequence to be generated.  Between 0 and infinity. Default to 0.
        do_sample: (`optional`) bool
            If set to `False` greedy decoding is used. Otherwise sampling is used. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.
        num_beams: (`optional`) int
            Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.
        temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        top_k: (`optional`) int
            The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
        top_p: (`optional`) float
            The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.
        repetition_penalty: (`optional`) float
            The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.
        pad_token_id: (`optional`) int
            Padding token. Default to specicic model pad_token_id or None if it does not exist.
        bos_token_id: (`optional`) int
            BOS token. Defaults to `bos_token_id` as defined in the models config.
        eos_token_id: (`optional`) int
            EOS token. Defaults to `eos_token_id` as defined in the models config.
        num_return_sequences: (`optional`) int
            The number of independently computed returned sequences for each element in the batch. Default to 1.
        attention_mask (`optional`) obj: `torch.LongTensor` of same shape as `input_ids`
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            Defaults to `None`.
        `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_start_token_id=None: (`optional`) int
            If an encoder-decoder model starts decoding with a different token than BOS.
            Defaults to `None` and is changed to `BOS` later.
    Return:
        output: `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`
            sequence_length is either equal to max_length or shorter if all batches finished early due to the `eos_token_id`
    Examples::
        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
        outputs = model.generate(max_length=40)  # do greedy decoding
        print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))
        tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
        input_context = 'The dog'
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
        for i in range(3): #  3 output sequences were generated
            print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))
        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
        input_context = 'The dog'
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
                    print((a[:, k_truncation:], b[:, :k_truncation]))
        outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3)  # 3 generate sequences using by sampling
        for i in range(3): #  3 output sequences were generated
            print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))
        tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
        input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
        print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))
        tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
        input_context = 'My cute dog'  # "Legal" is one of the control codes for ctrl
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True)  # generate sequences without allowing bad_words to be generated
    """

    # We cannot generate if the model does not have a LM head
    if model.get_output_embeddings() is None:
        raise AttributeError(
            "You tried to generate sequences with a model that does not have a LM Head."
            "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
        )

    assert schedule is not None
    assert isinstance(schedule, sampler.Sampler)
    max_length = max_length if max_length is not None else model.config.max_length
    do_sample = do_sample if do_sample is not None else model.config.do_sample
    num_beams = num_beams if num_beams is not None else model.config.num_beams
    assert num_beams==1
    temperature = temperature if temperature is not None else model.config.temperature
    repetition_penalty = repetition_penalty if repetition_penalty is not None else model.config.repetition_penalty
    assert repetition_penalty==1.0
    bos_token_id = bos_token_id if bos_token_id is not None else model.config.bos_token_id
    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else model.config.num_return_sequences
    )
    decoder_start_token_id = (
        decoder_start_token_id if decoder_start_token_id is not None else model.config.decoder_start_token_id
    )

    if input_ids is not None:
        batch_size = input_ids.shape[0]  # overriden by the input batch_size
    else:
        batch_size = 1

    assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
    assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
    assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
    assert temperature > 0, "`temperature` should be strictly positive."
    if not (isinstance(schedule, sampler.TemperatureSweep) or isinstance(schedule, sampler.KTemperatureSweep)):
        assert temperature==1.0
    assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
    assert input_ids is not None or (
        isinstance(bos_token_id, int) and bos_token_id >= 0
    ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
    assert pad_token_id is None or (
        isinstance(pad_token_id, int) and (pad_token_id >= 0)
    ), "`pad_token_id` should be a positive integer."
    assert (eos_token_id is None) or (
        isinstance(eos_token_id, int) and (eos_token_id >= 0)
    ), "`eos_token_id` should be a positive integer."
    assert (
        isinstance(num_return_sequences, int) and num_return_sequences > 0
    ), "`num_return_sequences` should be a strictly positive integer."

    if input_ids is None:
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
            "you should either supply a context to complete as `input_ids` input "
            "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
        )
        input_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=next(model.parameters()).device,
        )
    else:
        assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

    # not allow to duplicate outputs when greedy decoding
    if do_sample is False:
        raise Exception("Do Sample should never be false, are you still interpreting sampling?")
        if num_beams == 1:
            # no_beam_search greedy generation conditions
            assert (
                num_return_sequences == 1
            ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

        else:
            # beam_search greedy generation conditions
            assert (
                num_beams >= num_return_sequences
            ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

    # create attention mask if necessary
    # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
    if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
        attention_mask = input_ids.ne(pad_token_id).long()
    elif attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    # set pad_token_id to eos_token_id if not set. Important that this is done after
    # attention_mask is created
    if pad_token_id is None and eos_token_id is not None:
        logger.warning(
            "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
        )
        pad_token_id = eos_token_id

    # current position and vocab size
    vocab_size = model.config.vocab_size

    # set effective batch size and effective batch multiplier according to do_sample
    if do_sample:
        effective_batch_size = batch_size * num_return_sequences
        effective_batch_mult = num_return_sequences
    else:
        effective_batch_size = batch_size
        effective_batch_mult = 1

    if model.config.is_encoder_decoder:
        if decoder_start_token_id is None:
            decoder_start_token_id = bos_token_id

        assert (
            decoder_start_token_id is not None
        ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
        assert hasattr(model, "get_encoder"), "{} should have a 'get_encoder' function defined".format(model)
        assert callable(model.get_encoder), "{} should be a method".format(model.get_encoder)

        # get encoder and store encoder outputs
        encoder = model.get_encoder()

        encoder_outputs = encoder(input_ids, attention_mask=attention_mask)

    # Expand input ids if num_beams > 1 or num_return_sequences > 1
    if num_return_sequences > 1 or num_beams > 1:
        input_ids_len = input_ids.shape[-1]
        input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
        attention_mask = attention_mask.unsqueeze(1).expand(
            batch_size, effective_batch_mult * num_beams, input_ids_len
        )

        input_ids = input_ids.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
        attention_mask = attention_mask.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

    if model.config.is_encoder_decoder:
        # create empty decoder_input_ids
        input_ids = torch.full(
            (effective_batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        cur_len = 1

        assert (
            batch_size == encoder_outputs[0].shape[0]
        ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size)
            .view(-1, 1)
            .repeat(1, num_beams * effective_batch_mult)
            .view(-1)
            .to(input_ids.device)
        )
        # expand encoder_outputs
        encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

    else:
        encoder_outputs = None
        cur_len = input_ids.shape[-1]

    output, model_logits, next_token_logits = generate_no_beam_search(
        model=model,
        input_ids=input_ids,
        cur_len=cur_len,
        max_length=max_length,
        do_sample=do_sample,
        temperature=temperature,
        schedule=schedule,
        repetition_penalty=repetition_penalty,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        decoder_start_token_id=decoder_start_token_id,
        eos_token_id=eos_token_id,
        batch_size=effective_batch_size,
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        dry_run=dry_run,
    )

    return output, model_logits, next_token_logits


def generate_no_beam_search(
    model,
    input_ids,
    cur_len,
    max_length,
    do_sample,
    temperature,
    schedule,
    repetition_penalty,
    bos_token_id,
    pad_token_id,
    eos_token_id,
    decoder_start_token_id,
    batch_size,
    encoder_outputs,
    attention_mask,
    dry_run,
):
    """ Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
    """
    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(batch_size).fill_(1)
    sent_lengths = input_ids.new(batch_size).fill_(max_length)

    past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models
    all_model_logits = [] # initialize distribution over probabilities
    all_transformed_logits = [] # initialize transformed distribution over probabilities

    while cur_len < max_length:
        step = schedule.step()
        if isinstance(schedule, sampler.JointSampler):
            top_p = step['top_p']
            top_k = step['top_k']
        elif schedule.is_top_p:
            top_p = step
            top_k = None
        else:
            top_k = step
            top_p = None
        if dry_run:
            if schedule.is_top_p:
                print(f"Current Top P index: {top_p}")
            else:
                print(f"Current Top K index: {top_k}")

        model_inputs = model.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, use_cache=True)
        assert 'attention_mask' not in model_inputs
        # print(attention_mask)
        # print(model_inputs.keys())

        outputs = model(**model_inputs)
        all_model_logits.append(outputs[0].cpu().detach())
        next_token_logits = outputs[0][:, -1, :]

        # if model has past, then set the past variable to speed up decoding
        if model._use_cache(outputs, True):
            past = outputs[1]

        # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            model.enforce_repetition_penalty_(next_token_logits, batch_size, 1, input_ids, repetition_penalty)

        if do_sample:
            if isinstance(schedule, sampler.NoisedTemperatureSampler):
                #transform it before temperature
                next_token_logits = schedule.transform(next_token_logits)
 
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Check if we are using a joint sampler
            if isinstance(schedule, sampler.JointSampler):
                # do a min(top_p, top_k) 
                next_token_logits = combined_filtering(next_token_logits, top_k=top_k, top_p=top_p) 
            else:
                # cut off part of the distribution
                if isinstance(schedule, sampler.NegativeSampler):
                    assert not schedule.is_top_p
                    top_k, k_truncation = top_k
                    top_k_values, top_k_indicies = torch.topk(next_token_logits, k_truncation, dim=1)
                    if dry_run:
                        twice_top_k_values, _ = torch.topk(next_token_logits, k_truncation*2, dim=1)

                    least_value = torch.min(next_token_logits) - 1000
                    next_token_logits.scatter_(1, top_k_indicies, least_value)

                    if dry_run:
                        second_twice_top_k_values, _ = torch.topk(next_token_logits, k_truncation*2, dim=1)
                        assert (torch.all(twice_top_k_values[:, k_truncation:]==second_twice_top_k_values[:, :k_truncation]))
                    
                # Top-p/top-k/Simplex filtering
                least_value = torch.min(next_token_logits) - 1000
                if isinstance(schedule, sampler.UniformSimplexSampler) or isinstance(schedule, sampler.SortedSimplexSampler):
                    next_token_logits = schedule.transform(next_token_logits, temperature = temperature, least_value=least_value)
                elif isinstance(schedule, sampler.TargetEntropySampler) or isinstance(schedule, sampler.MaxEntropySampler) or isinstance(schedule, sampler.RandomSpaceTopkSampler):
                    assert(temperature == 1.0)
                    next_token_logits = schedule.transform(next_token_logits)
                elif isinstance(schedule, sampler.SortedNoisedFixedSampler):
                    next_token_logits = schedule.transform(next_token_logits)
                elif isinstance(schedule, sampler.NoisedTemperatureSampler):
                    pass
                    #we did the transform before the temperaturetuning
                    #next_token_logits = schedule.transform(next_token_logits)
                else:
                    assert(schedule, sampler.FixedSampler)
                    next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p, filter_value=least_value)

            all_transformed_logits.append(next_token_logits.unsqueeze(1).cpu().detach())
            probs = F.softmax(next_token_logits, dim=-1)
            if dry_run and (schedule.is_top_p or isinstance(schedule, sampler.JointSampler)):
                print("Number of tokens sampled from:", (probs > 0).sum(dim=1))
                print()
            
            """
            probs_ori = copy.deepcopy(probs)
            if isinstance(schedule, sampler.UniformSimplexSampler):
                for kk in range(probs.size(0)):
                    co = 0
                    while torch.sum(probs[kk]).item() > 1 - 1e-03:
                        probs[kk] = probs[kk] * (1 - 1e-3)
                        #print(co, end = ' ')
                        co = co + 1
                        #sys.stdout.flush()
            
            fn = 'tmp/logit_debug.save'
            torch.save({'probs': probs, 'probs_ori': probs_ori, 'logits': next_token_logits, 'time': str(datetime.datetime.now())}, fn)
            print(fn, datetime.datetime.now())
            time.sleep(0.1)
            """
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

        if eos_token_id is not None:
            eos_in_sents = tokens_to_add == eos_token_id
            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

        # extend attention_mask for new generated input if only decoder
        if model.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        cur_len = cur_len + 1

    # if there are different sentences lengths in the batch, some batches have to be padded
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
        # finished sents are filled with pad_token
        decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
    else:
        decoded = input_ids

    for hypo_idx, hypo in enumerate(input_ids):
        decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

    all_model_logits = torch.cat(all_model_logits, dim=1)
    all_transformed_logits = torch.cat(all_transformed_logits, dim=1)
    return decoded, all_model_logits, all_transformed_logits 

def combined_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > top_p
    cnt = sorted_indices_to_remove.sum(dim=1)
    ones = torch.ones_like(cnt)
    number_of_indicies_removed = torch.max(ones, cnt)
    indicies_remaining = sorted_logits.shape[-1] - number_of_indicies_removed + 1 # +1 to keep at least one token
    k = torch.ones_like(indicies_remaining) * top_k
    min_keep_tokens = torch.ones_like(k) * min_tokens_to_keep
    batchwise_top_k = torch.max(min_keep_tokens, torch.min(k, indicies_remaining)) 

    for idx in range(batchwise_top_k.shape[0]):
        top_k = batchwise_top_k[idx]
        top_k = min(max(top_k, min_tokens_to_keep), logits[idx].size(-1))  # Safety check

        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits[idx] < torch.topk(logits[idx], top_k)[0][..., -1, None]
        logits[idx][indices_to_remove] = filter_value

    return logits

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_p==None and top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    elif top_k==None and top_p <= 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def cut_seq_to_eos(sentence, remove_id=[-1]):
    sent=[]
    for s in sentence:
        if s in remove_id:
            continue
        if s != EOS_ID:
            sent.append(s)
        else:
            break
    return sent

