# greedy decoding

# first generate
# num_beams
# max_length : the maximum length of the generation for each prompt
# assisted_length: the maximum length of the generation from small speculative model


import argparse
import os
import time
from tqdm.auto import tqdm

import torch
import transformers
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    MBartTokenizer,
    MBartTokenizerFast,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM
)

import torch.nn.functional as F
import numpy as np

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from datasets import load_dataset

from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

import matplotlib.pyplot as plt

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="alpaca",
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--predict_with_generate",
        type=bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )

    parser.add_argument(
        "--assisted_length",
        type=int,
        default=16,
        help=(
            "Number of new generated tokens "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )

    parser.add_argument(
        "--small_temperature",
        type=float,
        default=1.0,
        help=(
            "softmax temperature"
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./output.log",
        # action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=False,
        help=(
            "Whether to pad all samples to model maximum sentence "
            "length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More"
            "efficient on GPU but very bad for TPU."
        ),
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument("--source_lang", type=str, default=None, help="Source language id for translation.")
    parser.add_argument("--target_lang", type=str, default=None, help="Target language id for translation.")
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=30,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--small_model_name_or_path",
        type=str,
        default="./LLaMA_MoE/",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--run",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--large_model_name_or_path",
        type=str,
        default="huggyllama/llama-7b",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--no_repeat",
        default=False,
        help="Whether allow repeat pattern in the generated sentence or not",
        action="store_true"
    )
    parser.add_argument(
        "--do_sample",
        default=False,
        help="Whether allow repeat pattern in the generated sentence or not",
        # action="store_true"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--first_generate",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="The path of the checkpoint whose state dict will be loaded into the model.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()
    # Sanity checks

    # if args.dataset_name is None and args.train_file is None and args.validation_file is None:
    #     raise ValueError("Need either a task name or a training/validation file.")
    #
    # if args.train_file is not None:
    #     extension = args.train_file.split(".")[-1]
    #     assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."...
    return args


class Evaluator:
    def __init__(self, dataset, tokenizer, accelerator, args=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.args = args

        # tokenize the dataset
        def tokenize_function(examples):
            if self.args.dataset_name == "prompt":
                example = self.tokenizer(examples['prompt'])
            elif self.args.dataset_name == "chatgpt":
                example = self.tokenizer(examples['human_prompt'])
            elif self.args.dataset_name == "webqa":
                example = self.tokenizer(examples['question'])
            elif self.args.dataset_name == "alpaca":
                example = self.tokenizer(examples['instruction'])
            elif self.args.dataset_name == "piqa":
                example = self.tokenizer(examples['goal'])
            else:
                example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, l_model, s_model, args):
        l_model.eval()
        s_model.eval()

        total_match = 0
        total_lrun = 0

        def _crop_past_key_values(model, past_key_values, maximum_length):
            """Crops the past key values up to a certain maximum length."""
            new_past = []
            if model.config.is_encoder_decoder:
                for idx in range(len(past_key_values)):
                    new_past.append(
                        (
                            past_key_values[idx][0][:, :, :maximum_length, :],
                            past_key_values[idx][1][:, :, :maximum_length, :],
                            past_key_values[idx][2],
                            past_key_values[idx][3],
                        )
                    )
                past_key_values = tuple(new_past)
            # bloom is special
            elif "bloom" in model.__class__.__name__.lower() or (
                    model.config.architectures is not None and "bloom" in model.config.architectures[0].lower()
            ):
                for idx in range(len(past_key_values)):
                    new_past.append(
                        (
                            past_key_values[idx][0][:, :, :maximum_length],
                            past_key_values[idx][1][:, :maximum_length, :],
                        )
                    )
                past_key_values = tuple(new_past)
            # gptbigcode is too
            elif "gptbigcode" in model.__class__.__name__.lower() or (
                    model.config.architectures is not None and "gptbigcode" in model.config.architectures[0].lower()
            ):
                if model.config.multi_query:
                    for idx in range(len(past_key_values)):
                        past_key_values[idx] = past_key_values[idx][:, :maximum_length, :]
                else:
                    for idx in range(len(past_key_values)):
                        past_key_values[idx] = past_key_values[idx][:, :, :maximum_length, :]
            else:
                for idx in range(len(past_key_values)):
                    new_past.append(
                        (
                            past_key_values[idx][0][:, :, :maximum_length, :],
                            past_key_values[idx][1][:, :, :maximum_length, :],
                        )
                    )
                past_key_values = tuple(new_past)
            return past_key_values

        requests_match = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.dataset)):
                per_match = 0
                per_lrun = 0
                orgin_len = len(batch['input_ids'])
                input_ids = batch['input_ids'].unsqueeze(0).cuda()
                kv_cache = None
                max_len = args.max_length + orgin_len

                while True:
                    cur_len = input_ids.shape[-1]
                    if args.num_beams > 1:
                        beam_outputs = s_model.generate(
                            input_ids,
                            max_length=args.first_generate + cur_len,
                            num_beams=args.num_beams,
                            # early_stopping=True,
                            return_dict_in_generate=False,
                            num_return_sequences=args.num_beams  # 返回所有 beams
                        )

                        idx = 0
                        match_len = -1
                        best_selected_tokens = None
                        best_kv_cache = None
                        for i, candidate_input_ids in enumerate(beam_outputs):
                            candidate_input_ids = candidate_input_ids.unsqueeze(0)
                            candidate_input_ids = s_model.generate(
                                candidate_input_ids,
                                max_length=args.assisted_length + cur_len,
                                # early_stopping=True,
                                return_dict_in_generate=False
                            )
                            candidate_input_ids = candidate_input_ids[0]
                            candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
                            if kv_cache is not None:
                                model_attn = torch.ones_like(candidate_input_ids)
                                model_input_ids = candidate_input_ids[:, -candidate_length - 1:]
                                outputs = l_model(
                                    model_input_ids,
                                    attention_mask=model_attn,
                                    past_key_values=kv_cache,
                                    use_cache=True,
                                )
                            else:
                                outputs = l_model(
                                    candidate_input_ids,
                                    use_cache=True,
                                )

                            new_logits = outputs.logits[:, -candidate_length - 1:]

                            if args.do_sample:
                                probs = new_logits[:, -candidate_length - 1:, :].softmax(dim=-1)
                                selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
                            else:
                                selected_tokens = new_logits[:, -candidate_length - 1:, :].argmax(dim=-1)

                            candidate_new_tokens = candidate_input_ids[:, -candidate_length:]
                            n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()
                            if n_matches > match_len:
                                match_len = n_matches
                                idx = i
                                best_selected_tokens = selected_tokens
                                best_kv_cache = outputs.past_key_values

                        n_matches = match_len
                        reduced = False
                        if beam_outputs[idx][-1] != self.tokenizer.eos_token_id and n_matches == candidate_length:
                            n_matches -= 1
                            reduced = True

                    else:
                        outputs = s_model.generate(
                            input_ids,
                            max_length=args.assisted_length + cur_len,
                            # num_beams=args.num_beams,
                            # early_stopping=True,
                            return_dict_in_generate=False,
                            # num_return_sequences=args.num_beams
                        )

                        candidate_input_ids = outputs[0]
                        candidate_input_ids = outputs[0].reshape(1,-1)  ##
                        candidate_length = candidate_input_ids.shape[-1] - input_ids.shape[-1]

                        if kv_cache is not None:
                            model_attn = torch.ones_like(candidate_input_ids)
                            model_input_ids = candidate_input_ids[:, -candidate_length - 1:]
                            outputs = l_model(
                                model_input_ids,
                                attention_mask=model_attn,
                                past_key_values=kv_cache,
                                use_cache=True,
                            )
                        else:
                            outputs = l_model(
                                candidate_input_ids,
                                use_cache=True,
                            )
                        new_logits = outputs.logits[:, -candidate_length - 1:]

                        if args.do_sample:
                            probs = new_logits[:, -candidate_length - 1:, :].softmax(dim=-1)
                            selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
                        else:
                            selected_tokens = new_logits[:, -candidate_length - 1:, :].argmax(dim=-1)
                        candidate_new_tokens = candidate_input_ids[:, -candidate_length:]
                        n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()
                        reduced = False
                        if candidate_input_ids.squeeze(0)[
                            -1] == self.tokenizer.eos_token_id and n_matches == candidate_length:
                            n_matches -= 1
                            reduced = True
                        best_selected_tokens = selected_tokens
                        best_kv_cache = outputs.past_key_values
                    n_matches = min(n_matches, max_len - cur_len - 1)

                    total_match += n_matches + 1
                    per_match += n_matches + 1
                    if reduced:
                        total_match += 1
                        per_match += 1
                    total_lrun += 1
                    per_lrun += 1

                    valid_tokens = best_selected_tokens[:, : n_matches + 1]
                    input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
                    new_cur_len = input_ids.shape[-1]
                    new_cache_size = new_cur_len - 1
                    kv_cache = _crop_past_key_values(l_model, best_kv_cache, new_cache_size)

                    if n_matches == int(args.assisted_length):
                        args.assisted_length += 2

                    if input_ids[0][-1] == self.tokenizer.eos_token_id or new_cur_len >= max_len:
                        break

                requests_match.append((per_match / per_lrun))

            with open(args.output_file, 'a') as file:
                print("-----------------------------------------------------------------------------------------------------------------")
                print(
                    f"Greedy Dataset-{args.dataset_name}, beam_size-{args.num_beams}, max_assisted_len-{args.assisted_length}, max_generate_len-{args.max_length}, average_assisted_tokens: {total_match / total_lrun}",
                    file=file)
            print(total_match / total_lrun)
            numpy_list = []
            for item in requests_match:
                if isinstance(item, torch.Tensor):
                    numpy_list.append(item.cpu().numpy())
                elif isinstance(item, float):
                    numpy_list.append(np.array(item))
                else:
                    pass
            np.save(f'{args.dataset_name} {args.num_beams} improved greedy {args.run}.npy', numpy_list)
            return total_match / total_lrun


def main():
    args = parse_args()

    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )

    transformers.logging.set_verbosity_error()

    if args.dataset_name == "openwebtext":
        eval_dataset = load_dataset('openwebtext', split='train[10000:11000]')
    elif args.dataset_name == "wikitext":
        eval_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[10000:11000]')
    elif args.dataset_name == "prompt":
        eval_dataset = load_dataset('alespalla/chatbot_instruction_prompts', split='train[:1000]')
    elif args.dataset_name == "chatgpt":
        eval_dataset = load_dataset('MohamedRashad/ChatGPT-prompts', split='train')
    elif args.dataset_name == "webqa":
        eval_dataset = load_dataset('web_questions', split='train[:1000]')
    elif args.dataset_name == "alpaca":
        eval_dataset = load_dataset('vicgalle/alpaca-gpt4', split='train[:100]')
    elif args.dataset_name == "piqa":
        eval_dataset = load_dataset('piqa', split='train[:1000]')

    #tokenizer = LlamaTokenizer.from_pretrained(args.large_model_name_or_path, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')
    l_model = LlamaForCausalLM.from_pretrained(args.large_model_name_or_path, device_map="auto")
    
    state_dict =  torch.load(args.checkpoint_path)
    s_moe_model = AutoModelForCausalLM.from_pretrained(args.small_model_name_or_path, \
                                                trust_remote_code=True, \
                                                state_dict=state_dict)
    s_moe_model = s_moe_model.cuda()
    evaluator = Evaluator(eval_dataset, tokenizer, accelerator, args)
    evaluator.evaluate(
        l_model=l_model,
        s_model=s_moe_model,
        args=args
    )


if __name__ == '__main__':
    main()
