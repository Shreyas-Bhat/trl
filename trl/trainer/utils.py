# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
# os.environ['OPENAI_API_KEY'] = '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'

import dataclasses
import importlib.resources as pkg_resources
import json
import random
import warnings
from collections import deque
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from openai import OpenAI
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from accelerate import Accelerator, PartialState
from accelerate.state import AcceleratorState
from huggingface_hub import ModelCard, ModelCardData
from rich.console import Console # type: ignore
from rich.table import Table # type: ignore
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    PreTrainedTokenizerBase,
    TrainerState,
    TrainingArguments,
)
from transformers.utils import (
    is_peft_available,
    is_torch_mlu_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

# torch.cuda.set_device(1)
from ..import_utils import is_unsloth_available
# from trl.import_utils import is_unsloth_available
from ..trainer.model_config import ModelConfig
# from trl.trainer.model_config import ModelConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
if is_peft_available():
    from peft import LoraConfig, PeftConfig # type: ignore


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://huggingface.co/papers/1909.08593
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index
        self.padding_free = padding_free

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        if self.padding_free:
            # remove padding, `attention_mask` and add `position_ids`
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index

        return batch


@dataclass
class DataCollatorForChatML:
    """
    Data collator for ChatML format datasets.
    """

    tokenizer: PreTrainedTokenizerBase
    ignore_index: int = -100
    max_length: int = None
    prompt_key: str = "prompt"
    messages_key: str = "messages"

    def __post_init__(self):
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer does not have a pad token. Please set `pad_token_id` in the tokenizer.")
        if self.max_length is None:
            # set a sensible default
            self.max_length = min(self.tokenizer.model_max_length, 1024)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        prompts_input_ids = []
        prompt_attention_mask = []
        labels = []

        for example in examples:
            formatted_prompt = example.get(self.prompt_key, None)
            if formatted_prompt is None:
                prompt = example[self.messages_key][:-1]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )

            if "input_ids" not in example:
                message = example[self.messages_key]
                formatted_message = self.tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                tokenized_message = self.tokenizer(
                    formatted_message,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=False,
                )
                input_ids.append(tokenized_message["input_ids"])
                attention_mask.append(tokenized_message["attention_mask"])
            else:
                input_ids.append(example["input_ids"])
                attention_mask.append(example["attention_mask"])

            tokenized_prompt = self.tokenizer(
                formatted_prompt,
                truncation=True,
                max_length=len(input_ids[-1]),
                padding=False,
                return_tensors=None,
                add_special_tokens=False,
            )

            prompts_input_ids.append(tokenized_prompt["input_ids"])
            prompt_attention_mask.append(tokenized_prompt["attention_mask"])

            # Create the labels that will have all but the completion tokens of the example["input_ids"] set to ignore_index
            label = [self.ignore_index] * len(input_ids[-1])
            completion_start_idx = len(tokenized_prompt["input_ids"])
            label[completion_start_idx:] = input_ids[-1][completion_start_idx:]
            labels.append(label)
            
        # convert to list of tensors and pad
        input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
        attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in attention_mask]
        labels = [torch.tensor(label, dtype=torch.long) for label in labels]
        input_ids = pad(input_ids, padding_side="left", padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad(attention_mask, padding_side="left", padding_value=0)
        labels = pad(labels, padding_side="left", padding_value=self.ignore_index)

        prompts_input_ids = [torch.tensor(ids, dtype=torch.long) for ids in prompts_input_ids]
        prompt_attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in prompt_attention_mask]
        prompts_input_ids = pad(prompts_input_ids, padding_side="left", padding_value=self.tokenizer.pad_token_id)
        prompt_attention_mask = pad(prompt_attention_mask, padding_side="left", padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompts": prompts_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
        }


@dataclass
class RewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        margin = []
        # check if we have a margin. If we do, we need to batch it as well
        has_margin = "margin" in features[0]
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
            if has_margin:
                margin.append(feature["margin"])
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        if has_margin:
            margin = torch.tensor(margin, dtype=torch.float)
            batch["margin"] = margin
        return batch


def pad(tensors: List[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`List[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


@dataclass
class DPODataCollatorWithPadding:
    r"""
    DPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.

    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        for k in features[0].keys():
            if k.endswith(("_input_ids", "_attention_mask", "_labels", "_pixel_values")):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.startswith(("chosen", "rejected", "completion")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # Set padding value based on the key
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                                " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                                " before calling the trainer."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.endswith("_pixel_values"):
                        padding_value = 0  # TODO: check if this is correct
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    # Set padding side based on the key
                    if k in ["prompt_input_ids", "prompt_attention_mask"]:
                        padding_side = "left"
                    else:
                        padding_side = "right"

                    # Set the dtype
                    if k.endswith("_pixel_values"):
                        dtype = torch.float32  # will be downcasted if necessary by the Trainer
                    else:
                        dtype = torch.int64

                    # Convert to tensor and pad
                    to_pad = [torch.tensor(ex[k], dtype=dtype) for ex in features]
                    padded_batch[k] = pad(to_pad, padding_value=padding_value, padding_side=padding_side)
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
    The dataset also formats the text before tokenization with a specific format that is provided
    by the user.

    Args:
        tokenizer (`transformers.PreTrainedTokenizer`):
            The processor used for processing the data.
        dataset (`dataset.Dataset`):
            Dataset with text files.
        dataset_text_field (`Optional[str]`, *optional*, defaults to `None`):
            Name of the field in the dataset that contains the text. Only one of `dataset_text_field` and
            `formatting_func` should be provided.
        formatting_func (`Callable`, *optional*):
            Function that formats the text before tokenization. Usually it is recommended to have follows a certain
            pattern such as `"### Question: {question} ### Answer: {answer}"`. Only one of `dataset_text_field` and
            `formatting_func` should be provided.
        infinite (`bool`, *optional*, defaults to `False`):
            If True the iterator is reset after dataset reaches end else stops.
        seq_length (`int`, *optional*, defaults to `1024`):
            Length of token sequences to return.
        num_of_sequences (`int`, *optional*, defaults to `1024`):
            Number of token sequences to keep in buffer.
        chars_per_token (`int`, *optional*, defaults to `3.6`):
            Number of characters per token used to estimate number of tokens in text buffer.
        eos_token_id (`int`, *optional*, defaults to `0`):
            Id of the end of sequence token if the passed tokenizer does not have an EOS token.
        shuffle (`bool`, *optional*, defaults to `True`)
            Shuffle the examples before they are returned
        append_concat_token (`bool`, *optional*, defaults to `True`)
            If true, appends `eos_token_id` at the end of each sample being packed.
        add_special_tokens (`bool`, *optional*, defaults to `True`)
            If true, tokenizers adds special tokens to each sample being packed.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        dataset_text_field=None,
        formatting_func=None,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        eos_token_id=0,
        shuffle=True,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        self.tokenizer = tokenizer

        if tokenizer.eos_token_id is None:
            warnings.warn(
                "The passed tokenizer does not have an EOS token. We will use the passed eos_token_id instead which corresponds"
                f" to {eos_token_id}. If this is not the correct EOS token, make sure to pass the correct eos_token_id."
            )

        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        self.append_concat_token = append_concat_token
        self.add_special_tokens = add_special_tokens

        if dataset_text_field is not None and formatting_func is not None:
            warnings.warn(
                "Only one of `dataset_text_field` and `formatting_func` should be provided. "
                "Ignoring `dataset_text_field` and using `formatting_func`."
            )

        if formatting_func is not None:
            self.formatting_func = formatting_func
        elif dataset_text_field is not None:
            self.formatting_func = lambda x: x[dataset_text_field]
        else:  # neither is provided
            raise ValueError("Either `dataset_text_field` or `formatting_func` should be provided.")

        if formatting_func is not None:
            if formatting_func.__code__.co_argcount > 1:
                warnings.warn(
                    "The passed formatting_func has more than one argument. Usually that function should have a single argument `example`"
                    " which corresponds to the dictionary returned by each element of the dataset. Make sure you know what you are doing."
                )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(self.formatting_func(next(iterator)))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    else:
                        more_examples = False
                        break
            if self.shuffle:
                random.shuffle(buffer)
            tokenized_inputs = self.tokenizer(buffer, add_special_tokens=self.add_special_tokens, truncation=False)[
                "input_ids"
            ]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                if self.append_concat_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                # Shuffle again, otherwise split examples occur in consecutive tensors.
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


@dataclass
class RunningMoments:
    """
    Calculates the running mean and standard deviation of a data stream. Reference:
    https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L75
    """

    accelerator: Accelerator
    mean: float = 0
    std: float = 1
    var: float = 1
    count: float = 1e-24

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks
        """
        if self.accelerator.use_distributed:
            xs_mean, xs_var, xs_count = get_global_statistics(self.accelerator, xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += (delta * xs_count / tot_count).item()
        new_var = tot_sum / tot_count
        self.std = (new_var * tot_count / (tot_count - 1)).float().sqrt().item()
        self.var = new_var.item()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        # save everything except accelerator
        if self.accelerator.is_main_process:
            save_dict = dataclasses.asdict(self, dict_factory=lambda x: {k: v for (k, v) in x if k != "accelerator"})
            json_string = json.dumps(save_dict, indent=2, sort_keys=True) + "\n"
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(json_string)

    @classmethod
    def load_from_json(cls, accelerator: Accelerator, json_path: str):
        """Create an instance from the content of `json_path`."""
        # load everything except accelerator
        with open(json_path, encoding="utf-8") as f:
            text = f.read()
        return cls(accelerator=accelerator, **json.loads(text))


@torch.no_grad()
def get_global_statistics(
    accelerator, xs: torch.Tensor, mask=None, device="cpu"
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Computes element-wise mean and variance of the tensor across processes. Reference:
    https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/utils.py#L57C1-L73C75
    """
    xs = xs.to(accelerator.device)
    sum_and_count = torch.tensor([xs.sum(), (xs.numel() if mask is None else mask.sum())], device=device)
    sum_and_count = accelerator.reduce(sum_and_count)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum(((xs - global_mean) ** 2).mul(1 if mask is None else mask))
    sum_var = accelerator.reduce(sum_var)
    global_var = sum_var / count

    return global_mean.to(device), global_var.to(device), count.item()


def compute_accuracy(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    # Here, predictions is rewards_chosen and rewards_rejected.
    # We want to see how much of the time rewards_chosen > rewards_rejected.
    if np.array(predictions[:, 0] == predictions[:, 1], dtype=float).sum() > 0:
        warnings.warn(
            f"There are {np.array(predictions[:, 0] == predictions[:, 1]).sum()} out of {len(predictions[:, 0])} instances where the predictions for both options are equal. As a consequence the accuracy can be misleading."
        )
    predictions = np.argmax(predictions, axis=1)

    accuracy = np.array(predictions == labels, dtype=float).mean().item()
    return {"accuracy": accuracy}


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=device),
            ],
            dim=dim,
        )


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def exact_div(a, b, custom_error_message=""):
    q = a // b
    if a != q * b:
        raise ValueError(f"{custom_error_message}, inexact division: {a} / {b} = {a / b}")
    return q


# copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/stat_tracking.py#L5
class PerPromptStatTracker:
    r"""
    Class for tracking statistics per prompt. Mainly used to calculate advantage for the DPPO algorithm

    Args:
        buffer_size (`int`):
            Size of the buffer to keep for each prompt.
        min_count (`int`):
            Minimum number of samples to keep in the buffer before calculating the mean and std.
    """

    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def update(self, prompts, rewards):
        prompts = np.array(prompts)
        rewards = np.array(rewards)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = deque(maxlen=self.buffer_size)
            self.stats[prompt].extend(prompt_rewards)

            if len(self.stats[prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[prompt])
                std = np.std(self.stats[prompt]) + 1e-6
            advantages[prompts == prompt] = (prompt_rewards - mean) / std

        return advantages

    def get_stats(self):
        return {k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)} for k, v in self.stats.items()}


def peft_module_casting_to_bf16(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
        elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    # module = module.to(torch.bfloat16)
                    module = module.to(torch.bfloat8)


def trl_sanitze_kwargs_for_tagging(model, tag_names, kwargs=None):
    if is_unsloth_available():
        # Unsloth adds a new attribute in the model config `unsloth_version`
        # to keep track of models that have been patched with unsloth.
        if hasattr(model, "config") and getattr(model.config, "unsloth_version", None) is not None:
            tag_names.append("unsloth")

    if kwargs is not None:
        if "tags" not in kwargs:
            kwargs["tags"] = tag_names
        elif "tags" in kwargs and isinstance(kwargs["tags"], list):
            kwargs["tags"].extend(tag_names)
        elif "tags" in kwargs and isinstance(kwargs["tags"], str):
            tag_names.append(kwargs["tags"])
            kwargs["tags"] = tag_names
    return kwargs


def get_quantization_config(model_config: ModelConfig) -> Optional[BitsAndBytesConfig]:
    if model_config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_config.torch_dtype,  # For consistency with model weights, we use the same value as `torch_dtype`
            bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_config.use_bnb_nested_quant,
            bnb_4bit_quant_storage=model_config.torch_dtype,
        )
    elif model_config.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_kbit_device_map() -> Optional[Dict[str, int]]:
    if is_torch_xpu_available():
        return {"": f"xpu:{PartialState().local_process_index}"}
    elif torch.cuda.is_available():
        return {"": PartialState().local_process_index}
    else:
        return None


def get_peft_config(model_config: ModelConfig) -> "Optional[PeftConfig]":
    if model_config.use_peft is False:
        return None

    if not is_peft_available():
        raise ValueError(
            "You need to have PEFT library installed in your environment, make sure to install `peft`. "
            "Make sure to run `pip install -U peft`."
        )

    peft_config = LoraConfig(
        task_type=model_config.lora_task_type,
        r=model_config.lora_r,
        target_modules=model_config.lora_target_modules,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        bias="none",
        use_rslora=model_config.use_rslora,
        modules_to_save=model_config.lora_modules_to_save,
    )

    return peft_config


def get_exp_cap(value, decimal=4):
    """
    Get the exponent cap of a value. This is used to cap the exponent of a value to avoid overflow.
    The formula is : log(value.dtype.max)
    E.g.
      For float32 data type, the maximum exponent value is 88.7228 to 4 decimal points.
    ```

    Args:
        value (`torch.Tensor`):
            The input tensor to obtain the data type
        decimal (`int`):
            The number of decimal points of the output exponent cap.
            eg: direct calling exp(log(torch.float32.max)) will result in inf
            so we cap the exponent to 88.7228 to avoid overflow.
    """
    vdtype_max = torch.zeros([1]).to(value.dtype) + torch.finfo(value.dtype).max
    vdtype_log_max = torch.log(vdtype_max).to(device)
    return torch.floor(vdtype_log_max * 10**decimal) / 10**decimal if decimal > 0 else vdtype_log_max


def cap_exp(value, cap=-1):
    # Cap the exponent value below the upper-bound to avoid overflow, before calling torch.exp
    cap = get_exp_cap(value) if cap < 0 else cap
    return torch.exp(torch.clamp(value, max=cap))


def print_rich_table(df: pd.DataFrame) -> Table:
    console = Console()
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.print(table)


SIMPLE_SFT_CHAT_TEMPLATE = "{% for message in messages %}{{' ' + message['content']}}{% endfor %}{{ eos_token }}"
# SIMPLE_SFT_CHAT_TEMPLATE simply ends things with an EOS token, this helps the SFT model learn to end the completions with EOS tokens

SIMPLE_CHAT_TEMPLATE = "{% for message in messages %}{{message['role'].capitalize() + ': ' + message['content'] + '\n\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"


@dataclass
class OnlineTrainerState(TrainerState):
    episode: int = 0


@dataclass
class OnPolicyConfig(TrainingArguments):
    r"""
    Base configuration class for on-policy trainers.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        run_name (`Optional[str]`, *optional*, defaults to `None`):
            Name of the run.
        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        num_mini_batches (`int`, *optional*, defaults to `1`):
            Number of minibatches to split a batch into.
        total_episodes (`Optional[int]`, *optional*, defaults to `None`):
            Total number of episodes in the dataset.
        local_rollout_forward_batch_size (`int`, *optional*, defaults to `64`):
            Per rank no grad forward pass in the rollout phase.
        num_sample_generations (`int`, *optional*, defaults to `10`):
            Number of debugging samples generations (i.e., `generate_completions` calls) throughout training.
        response_length (`int`, *optional*, defaults to `53`):
            Length of the response.
        stop_token (`Optional[str]`, *optional*, defaults to `None`):
            Stop token.
        stop_token_id (`Optional[int]`, *optional*, defaults to `None`):
            Truncation token id.
        temperature (`float`, *optional*, defaults to `0.7`):
            Sampling temperature.
        missing_eos_penalty (`Optional[float]`, *optional*, defaults to `None`):
            Penalty applied to the score when the model fails to generate an EOS token. This is useful to encourage
            to generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be a positive
            value.
        sft_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the SFT model.
        world_size (`Optional[int]`, *optional*, defaults to `None`):
            Number of processes (GPUs) to use for the training.
        num_total_batches (`Optional[int]`, *optional*, defaults to `None`):
            Number of total batches to train.
        micro_batch_size (`Optional[int]`, *optional*, defaults to `None`):
            Micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`).
        local_batch_size (`Optional[int]`, *optional*, defaults to `None`):
            Batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`).
        batch_size (`Optional[int]`, *optional*, defaults to `None`):
            Batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`).
        local_mini_batch_size (`Optional[int]`, *optional*, defaults to `None`):
            Mini batch size per GPU.
        mini_batch_size (`Optional[int]`, *optional*, defaults to `None`):
            Mini batch size across GPUs.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the model to the Hub after training.
    """

    run_name: Optional[str] = None
    dataset_num_proc: Optional[int] = None
    num_mini_batches: int = 1
    total_episodes: Optional[int] = None
    local_rollout_forward_batch_size: int = 64
    num_sample_generations: int = 10
    # response_length: int = 512
    response_length: int = 128
    stop_token: Optional[Literal["eos"]] = None
    stop_token_id: Optional[int] = None
    temperature: float = 0.7
    missing_eos_penalty: Optional[float] = None
    sft_model_path: str = "EleutherAI/pythia-160m"
    world_size: Optional[int] = None
    num_total_batches: Optional[int] = None
    micro_batch_size: Optional[int] = None
    local_batch_size: Optional[int] = None
    batch_size: Optional[int] = None
    local_mini_batch_size: Optional[int] = None
    mini_batch_size: Optional[int] = None
    push_to_hub: bool = False


def first_true_indices(bools: torch.Tensor, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.

    Args:
        bools (`torch.Tensor`):
            An N-dimensional boolean tensor.
        dtype (`torch.dtype`, optional):
            The desired data type of the output tensor. Defaults to `torch.long`.

    Returns:
        `torch.Tensor`:
            An (N-1)-dimensional tensor of integers indicating the position of the first True
            in each row. If no True value is found in a row, returns the length of the row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=device)
    return torch.min(zero_or_index, dim=-1).values

def get_reward(
    model: torch.nn.Module, 
    query_responses: torch.Tensor, 
    pad_token_id: int, 
    context_length: int, 
    llm_scores: torch.Tensor, 
    ground_truth: torch.Tensor, 
    tokenizer=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the reward using LLM sentiment classification.
    Returns 0.95 for positive sentiment, 0.05 for negative sentiment.
    """
    attention_mask = query_responses != tokenizer.pad_token_id
    # query_responses = query_responses[attention_mask]
    # query_responses = query_responses[query_responses != tokenizer.eos_token_id]
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    # Convert tensor to text for LLM processing
    texts = []
    # client = OpenAI()
    summary_prompts = []
    generated_texts = []
    for i in range(query_responses.shape[0]):
        # valid_tokens = query_responses[i][context_length:]
        valid_tokens = query_responses[i]



        # [attention_mask[i]]
        if tokenizer:
            text = tokenizer.decode(valid_tokens)
            # system_message = """
            # You are an expert sentiment analyst of movies.
            # """
                
            # prompt = f"""You are an expert sentiment analyst of movies. Given {text}, analyze and state whether the movie is likely to be "Positive" or "Negative". Provide the label exactly as one of the following:\n\n"
            # "<label>High confidence "Negative"</label> \n"
            # "<label>Moderate confidence "Negative"</label> \n"
            # "<label>Low confidence "Negative"</label> \n"
            # "<label>High confidence "Positive"</label> \n"
            # "<label>Moderate confidence "Positive"</label> \n"
            # "<label>Low confidence "Positive"</label> \n"
            # "Do not include any additional formatting or characters, just return the label within the <label></label> tags."""
            prompt = f"""{text}"""


            summary_prompts.append(prompt)
            texts.append(prompt)
    # for i in range(query_responses.shape[0]):
    #     valid_tokens = query_responses[i]
        
        # if tokenizer:
        #     text = tokenizer.decode(valid_tokens)
            
        #     # Prepare messages for OpenAI API
        #     messages = [
        #         {
        #             "role": "system",
        #             "content": """You are an expert sentiment analyst. Your task is to:
        #             1. Read the provided text
        #             2. Classify it as either positive or negative
        #             3. Reply ONLY with the word 'Positive' or 'Negative'"""
        #         },
        #         {
        #             "role": "user",
        #             "content": f"Text: {text}\nClassification:"
        #         }
        #     ]
            
        #     # Store both original text and formatted messages
        #     texts.append(text)  # Store original decoded text
        #     summary_prompts.append(messages)
    #         print("texts", texts)
    # # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # # tokenizer.padding_side = 'left'
    # # device = next(model.parameters()).device
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    # model.eval()
    with torch.no_grad():
        # Generate complete responses
        # generated_outputs = model.bfloat16().generate(
        #     **inputs,
        #     # inputs["input_ids"],
        #     max_new_tokens=200,  # Adjust based on expected response length
        #     num_beams=1,       # Use greedy decoding
        #     do_sample=True,   # Don't use sampling
        #     pad_token_id=tokenizer.pad_token_id,
        #     eos_token_id=tokenizer.eos_token_id,
        #     return_dict_in_generate=True,
        #     output_scores=True  # Get the scores for each token
        # )
        
        generated_outputs = model.generate(
            **inputs,
            remove_invalid_values=True,
            # inputs["input_ids"],
            max_new_tokens=50,  # Adjust based on expected response length
            min_length = 2,
            num_beams=1,       # Use greedy decoding
            do_sample=True,   # Don't use sampling
            pad_token_id=tokenizer.pad_token_id,
            top_k = 50,
            top_p = 0.9,
            temperature = 0.1, 
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            # suppress_tokens=[tokenizer.eos_token_id],
            output_scores=True  # Get the scores for each token
        )
        # Decode the generated sequences
        generated_texts = []
        for output_ids in generated_outputs.sequences:
            # Get only the newly generated tokens (exclude input prompt)
            new_tokens = output_ids[inputs['input_ids'].shape[1]:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            generated_texts.append(generated_text)
            
        # print("\nGenerated responses:")
        # for idx, (prompt, response) in enumerate(zip(texts, generated_texts)):
        #     print(f"Input {idx}: {prompt}")
        #     print(f"Output: {response}")
        #     print("-" * 50)
    # for text in texts:
    #     system_message = {
    #         "role": "system",
    #         "content": """You are an expert sentiment analyst. Your task is to:
    #         1. Read the provided text
    #         2. Classify it as either positive or negative
    #         3. Reply ONLY with the word 'Positive' or 'Negative'"""
    #     }
        
    #     user_message = {
    #         "role": "user",
    #         "content": f"Text: {text}\nClassification:"
    #     }
        
    #     summary_prompts.append([system_message, user_message])

    # # Process each prompt with API
    # for messages in summary_prompts:
        
    #     completion = client.chat.completions.create(
    #         model="gpt-4o-mini",  # or your preferred model
    #         messages=messages,
    #         temperature=0.1,
    #         max_tokens=200,
    #         top_p=0.9,
    #         frequency_penalty=0,
    #         presence_penalty=0
    #     )
            
    #         # Extract response
    #     response = completion.choices[0].message.content
    #     generated_texts.append(response.strip())
    #     # Convert responses to binary probabilities
    # llm_probabilities = torch.zeros(ground_truth.shape[0]).to(device)
    # for i, text in enumerate(generated_texts):
    #     # text = text.lower().strip()
    #     print("Prediction output: \n", text)
    #     # You might need to adjust this based on actual outputs
    #     if 'Output: Negative' in text:
    #         llm_probabilities[i] = 0.20
    #     elif 'Output: Positive' in text:
    #         llm_probabilities[i] = 0.80
        # positive_id = tokenizer.encode(' positive', add_special_tokens=False)[0]
        # negative_id = tokenizer.encode(' negative', add_special_tokens=False)[0]

        # # Initialize probabilities tensor
        llm_probabilities = torch.zeros(len(generated_texts)).to(device)
        for i, text in enumerate(generated_texts):
            text = text.lower()
            
            # Check for sentiment and confidence level
            if "negative" in text:
                if "high confidence" in text:
                    llm_probabilities[i] = 0.05
                elif "moderate" in text:
                    llm_probabilities[i] = 0.2
                elif "low" in text:
                    llm_probabilities[i] = 0.4
                else:
                    llm_probabilities[i] = 0.15
            elif "positive" in text:
                if "high" in text:
                    llm_probabilities[i] = 0.95
                elif "moderate" in text:
                    llm_probabilities[i] = 0.8
                elif "low" in text:
                    llm_probabilities[i] = 0.6
                else:
                    llm_probabilities[i] = 0.7
        # for i, text in enumerate(generated_texts):
        #     text = text.lower().strip()
        #     print(f"\nText {i}: {text}")
        #     # Find the position of "positive" or "negative" in the generated text
        #     token_ids = tokenizer.encode(text, add_special_tokens=False)
        #     tokens = tokenizer.convert_ids_to_tokens(token_ids)
        #     print("Token IDs:", token_ids)
        #     print("Tokens:", tokens)
        #     # pos_tokens = tokenizer.encode('positive', add_special_tokens=False)
        #     # neg_tokens = tokenizer.encode('negative', add_special_tokens=False)
        #     # print("'positive' token ids:", pos_tokens)
        #     # print("'negative' token ids:", neg_tokens)
        #     # print("'positive' tokens:", tokenizer.convert_ids_to_tokens(pos_tokens))
        #     # print("'negative' tokens:", tokenizer.convert_ids_to_tokens(neg_tokens))
        # pos_variants = [
        #     tokenizer.encode('positive', add_special_tokens=False)[0],
        #     tokenizer.encode(' positive', add_special_tokens=False)[0],
        #     tokenizer.encode('Positive', add_special_tokens=False)[0],
        #     tokenizer.encode(' Positive', add_special_tokens=False)[0]
        # ]
        # neg_variants = [
        #     tokenizer.encode('negative', add_special_tokens=False)[0],
        #     tokenizer.encode(' negative', add_special_tokens=False)[0],
        #     tokenizer.encode('Negative', add_special_tokens=False)[0],
        #     tokenizer.encode(' Negative', add_special_tokens=False)[0]
        # ]

        #     # # Remove duplicates and print for debugging
        #     # pos_variants = list(set(pos_variants))
        #     # neg_variants = list(set(neg_variants))
        #     # print("Positive token variants:", pos_variants)
        #     # print("Negative token variants:", neg_variants)

        #     # llm_probabilities = torch.zeros(len(generated_texts)).to(device)

        # for i, text in enumerate(generated_texts):
        #     text = text.lower().strip()
        #     print("text", text)
        #     token_ids = tokenizer.encode(text, add_special_tokens=False)
        #     print(token_ids, pos_variants)
        #     # Check for any variant of positive
        #     pos_positions = [i for i, t in enumerate(token_ids) if t in pos_variants]
        #     if pos_positions:
        #         pos = pos_positions[0]  # Take first occurrence
        #         # Get logits for that position
        #         logits = generated_outputs.scores[pos][i]
                
        #         # Apply softmax to convert logits to probabilities
        #         probs = torch.softmax(logits, dim=-1)
                
        #         # Get probabilities for positive variants
        #         pos_probs = torch.tensor([probs[token_id] for token_id in pos_variants])
        #         prob = torch.max(pos_probs)  # Get highest probability among positive tokens
        #         llm_probabilities[i] = prob.item()
            
        #     # Check for any variant of negative
        #     neg_positions = [i for i, t in enumerate(token_ids) if t in neg_variants]
        #     if neg_positions:
        #         pos = neg_positions[0]  # Take first occurrence
        #         # Get logits for that position
        #         logits = generated_outputs.scores[pos][i]
                
        #         # Apply softmax to convert logits to probabilities
        #         probs = torch.sigmoid(logits, dim=-1)
                
        #         # Get probabilities for positive variants
        #         pos_probs = torch.tensor([probs[token_id] for token_id in neg_variants])
        #         prob = torch.max(pos_probs)  # Get highest probability among positive tokens
        #         llm_probabilities[i] = 1.0 - prob.item()

        # Reshape and calculate CE
    # positive_words = {'good', 'great', 'excellent', 'amazing', 'love', 'awesome', 
    #             'fantastic', 'wonderful', 'best', 'brilliant', 'enjoyed'}
   

    # llm_probabilities = torch.zeros(len(generated_texts)).to(device) 
    # # Integration with existing code:
    # for i, text in enumerate(generated_texts):
    #     words = text.lower().split()
    #     pos_count = sum(1 for word in words if word in positive_words)
        
    #     # If has positive words, reward is 0.8, else 0.2
    #     if pos_count > 0:
    #         reward = 0.8
    #     else:
    #         reward = 0.2
    #     llm_probabilities[i] = reward
    llm_probabilities = llm_probabilities.view(-1, 1)
    ground_truth = ground_truth.float().view(-1, 1)
    # adjusted_probabilities = torch.where(
    # ground_truth == 1,
    # llm_probabilities,
    # 1 - llm_probabilities
    # )

    # # Calculate cross-entropy with adjusted probabilities
    # epsilon = 1e-7
    # cross_entropy = -torch.log(adjusted_probabilities.abs() + epsilon)


    epsilon = 1e-7
    cross_entropy = -ground_truth * torch.log(llm_probabilities.abs() + epsilon) - \
                (1 - ground_truth) * torch.log(1 - llm_probabilities.abs() + epsilon)
    metrics_data = []
    for i in range(len(generated_texts)):
        token_ids = tokenizer.encode(generated_texts[i], add_special_tokens=False)
        metrics_dict = {
            'input_text': texts[i],
            'model_output': generated_texts[i],
            'LLM probabilites': llm_probabilities[i].item(),
            'Ground Truth': ground_truth[i].item(),
            'cross_entropy': cross_entropy[i].item() if cross_entropy.dim() > 0 else cross_entropy.item(),
            'tokens': token_ids,
            # 'pos variants': pos_variants,
            'token_text': [tokenizer.decode([t]) for t in token_ids],
            # 'neg variants': neg_variants
        }
        metrics_data.append(metrics_dict)
    df = pd.DataFrame(metrics_data)
    filename = f'rewards.csv'
    # If file exists, append without headers
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        # If file doesn't exist, create new with headers
        df.to_csv(filename, mode='w', header=True, index=False)
    print("\nMetrics:")
    print("Ground truth:", ground_truth)
    print("LLM probabilities:", llm_probabilities)
    print("Cross entropy:", cross_entropy)
    
    return (
        llm_probabilities,
        cross_entropy,
        sequence_lengths,
    )
def forward(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    pad_token_id: int,
) -> torch.nn.Module:
    """
    Performs a forward pass through the model with the given query responses and pad token ID.

    Args:
        model (`torch.nn.Module`):
            The model to perform the forward pass.
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.

    Returns:
        `torch.nn.Module`:
            The output of the model, including hidden states.
    """
    attention_mask = query_responses != pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def prepare_deepspeed(
    model: torch.nn.Module, per_device_train_batch_size: int, fp16: bool = False, bf16: bool = False
):
    """
    Prepares the model for training with DeepSpeed (both for stage 2 and 3), configuring the appropriate settings based on the model and
    batch size.

    Args:
        model (`torch.nn.Module`):
            The model to be prepared for DeepSpeed training.
        per_device_train_batch_size (`int`):
            The training batch size per device.

    Returns:
        `torch.nn.Module`:
            The model initialized and configured with DeepSpeed for training.
    """
    import deepspeed

    deepspeed_plugin = AcceleratorState().deepspeed_plugin
    config_kwargs = deepspeed_plugin.deepspeed_config
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["train_micro_batch_size_per_gpu"] = per_device_train_batch_size
        config_kwargs = {
            "train_micro_batch_size_per_gpu": config_kwargs["train_micro_batch_size_per_gpu"],
            "prescale_gradients": False,
            "wall_clock_breakdown": False,
        }
        if bf16:
            config_kwargs["bf16"] = {"enabled": True}
        elif fp16:
            config_kwargs["fp16"] = {"enabled": True}
    else:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0,
                    }
                )
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


def truncate_response(stop_token_id: int, pad_token_id: int, responses: torch.Tensor):
    """
    Truncates the responses at the first occurrence of the stop token, filling the rest with pad tokens.

    Args:
        stop_token_id (`int`):
            The token ID representing the stop token where truncation occurs.
        pad_token_id (`int`):
            The token ID representing the pad token used to fill the truncated responses.
        responses (`torch.Tensor`):
            The tensor containing the responses to be truncated.

    Returns:
        `torch.Tensor`:
            The truncated responses tensor with pad tokens filled after the stop token.
    """
    trunc_idxs = first_true_indices(responses == stop_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, pad_token_id)
    return postprocessed_responses


def generate(
    lm_backbone: torch.nn.Module, queries: torch.Tensor, pad_token_id: int, generation_config: GenerationConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone in a way that does not affect padding tokens.

    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        pad_token_id (`int`):
            The token ID representing the pad token.
        generation_config (`GenerationConfig`):
            The configuration for the generation process.

    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    # print("Queries shape:", queries.shape)
    # print("Queries device:", queries.device)
    # print("Unique token IDs in queries:", torch.unique(queries).tolist())
    # print("Pad token ID:", pad_token_id)
    # print("Generation config:", vars(generation_config))
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    # attention_mask = attention_mask.to(device)
    # print("Number of padding tokens:", (queries == pad_token_id).sum().item())
    # print("Number of non-padding tokens:", (queries != pad_token_id).sum().item())
    # print("Generation config pad_token_id:", generation_config.pad_token_id)
    input_ids = torch.masked_fill(queries, ~attention_mask, 0).to(device)
    # print("Checking shapes:", input_ids.shape, attention_mask.shape)
    # print("Checking type:", input_ids.dtype, attention_mask.dtype)
    # print("Checking device:", input_ids.device, attention_mask.device)
    # print("input_ids", input_ids)
    original_texts = []
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.padding_side = "left"
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})\
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.padding_side = 'left'
    for i in range(queries.shape[0]):
        valid_tokens = input_ids[i][attention_mask[i]]
        text = tokenizer.decode(valid_tokens, skip_special_tokens=True)
        original_texts.append(text)
    summary_prompts = []
    for text in original_texts:
        # system_message = "You are an expert sentiment analyst skilled in determining whether text is positive or negative. Reply only with 'Positive' or 'Negative'."
        
        # prompt = f"{system_message}. Follow this same template when generating outputs and here is the text to analyze:\n\n{text}\n\nSentiment:"
        prompt = f"""You are an expert sentiment analyst of movies. Given {text}, analyze and state whether the movie is likely to be "Positive" or "Negative". Provide the label exactly as one of the following:\n\n"
            "<label>High confidence "Negative"</label> \n"
            "<label>Moderate confidence "Negative"</label> \n"
            "<label>Low confidence "Negative"</label> \n"
            "<label>High confidence "Positive"</label> \n"
            "<label>Moderate confidence "Positive"</label> \n"
            "<label>Low confidence "Positive"</label> \n"
            "Do not include any additional formatting or characters, just return the label within the <label></label> tags."""
        summary_prompts.append(prompt)
        # summary_prompts.append(prompt)
    # print("summary prompts", summary_prompts)
    # breakpoint()
    summary_inputs = tokenizer(
        summary_prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128
    ).to(device)
    
    # with torch.cuda.amp.autocast(True):[]
    # Check your inputs first
    # print("Input shape:", summary_inputs['input_ids'].shape)
    # print("Attention mask shape:", summary_inputs['attention_mask'].shape)
    # print("Sample of attention mask:", summary_inputs['attention_mask'][0][:50])

    # Add checks for special tokens
    # print("First token:", summary_inputs['input_ids'][0][0])  # Should be BOS token if used
    # print("Sample sequence:", summary_inputs['input_ids'][0][:50])

    # Verify no empty sequences
    # print("Any empty sequences?", (summary_inputs['attention_mask'].sum(dim=1) == 0).any())

    # Check for numerical issues
    # print("Device:", summary_inputs['input_ids'].device)
    # print("Dtype:", lm_backbone.dtype)

    # You might want to try fp32 instead of bfloat16 to test if it's a precision issue
    # test_output = lm_backbone.float().generate(...)
    torch.cuda.empty_cache()
    # lm_backbone.eval()
    # weights = torch.cat([p.data.flatten() for p in lm_backbone.parameters()])
    # print(weights.shape)
    # summary_output = lm_backbone.generate( #I changed sampling to False here 
    #     **summary_inputs,  # Unpack all inputs including input_ids and attention_mask
    #     # num_beams=1,
    #     # remove_invalid_values=True,
    #     do_sample=True,
    #     generation_config=generation_config,
    #     return_dict_in_generate=True,
    #     output_scores=True,
        
    # )
    generation_config = GenerationConfig(
        max_new_tokens=200,
        temperature=0.1,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        min_length = 2,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        suppress_tokens=[tokenizer.eos_token_id],
        output_scores=True,
    )
    summary_output = lm_backbone.generate(
            **summary_inputs,
            generation_config=generation_config
        )
    vocab_size = lm_backbone.config.vocab_size
    vocab_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
    if tokenizer.pad_token_id is not None:
        vocab_mask[tokenizer.pad_token_id] = False
    # print("Generation config used:", summary_output.generation_config)
    # summary_output = lm_backbone.bfloat16().cuda().generate(
    #     # input_ids=input_ids,
    #     input_ids=summary_inputs.input_ids,
    #     attention_mask=summary_inputs.attention_mask,
    #     # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # not needed: already adjusted in generations
    #     # https://github.com/huggingface/transformers/blob/ac33aeeeee2a7a89b89c93c2962e6feb90daef0a/src/transformers/models/gpt2/modeling_gpt2.py#L1227-L1250
    #     generation_config=generation_config,
    #     return_dict_in_generate=True,
    #     output_scores=True,
    # )
    # print("summary output", summary_output)
    summary_texts = []

    
    for summary_ids in summary_output.sequences:
        summary = tokenizer.decode(
            # summary_ids[summary_inputs.input_ids.shape[1]:],
            summary_ids[summary_inputs['input_ids'].shape[1]:],
            skip_special_tokens = True
        ).strip()
        # print("summary follsz:", summary)
        summary_texts.append(summary)
    processed_scores = []
    for i, score in enumerate(summary_output.scores):
        # Get logits
        logits = torch.where(
            torch.isinf(score) & (score < 0),
            torch.tensor(-0.001, device=score.device, dtype=score.dtype),
            score
        )
        # print(f"\nStep {i} Analysis:")
        # print("Raw logits stats:")
        # print(f"Contains -inf: {torch.isinf(logits).any().item()}")
        # print(f"Min logit: {logits.min().item()}")
        # print(f"Max logit: {logits.max().item()}")
        
        # Apply temperature
        scaled_logits = logits
        
        # Create attention mask for valid tokens
        vocab_mask = torch.ones_like(logits, dtype=torch.bool)
        vocab_mask[:, tokenizer.pad_token_id] = False  # Mask pad token
        
        # Apply masked softmax
        masked_logits = torch.where(
            vocab_mask,
            scaled_logits,
            torch.tensor(-1e4, device=logits.device, dtype=logits.dtype)
        )
        probs = torch.softmax(masked_logits, dim=-1)
        
        # Compute log probabilities
        log_probs = torch.log(probs + 1e-10)  # Add small epsilon to prevent -inf
        
        processed_scores.append(log_probs)

    logits = torch.stack(processed_scores, 1)
    # for score in summary_output.scores:
    #     # Expand mask to match score shape
    #     expanded_mask = vocab_mask.expand_as(score)
    #     # Replace -inf with large negative number where vocab mask is False
    #     # masked_score = torch.where(
    #     #     torch.isinf(score) & (score < 0),  # Find -inf values
    #     #     torch.tensor(-1e4, device=score.device, dtype=score.dtype),
    #     #     score
    #     # )
    #     non_inf_values = score[~torch.isinf(score)]
    #     # processed_scores.append(masked_score)
    #     processed_scores.append(non_inf_values)
    # logits = torch.stack(processed_scores, 1)
    # logits = torch.stack(summary_output.scores, 1)
    torch.cuda.empty_cache()
    return torch.cat((queries, summary_output.sequences[:, context_length:]), dim=1), logits


@torch.no_grad()
def batch_generation(
    model: torch.nn.Module,
    queries: torch.Tensor,
    local_rollout_forward_batch_size: int,
    pad_token_id: int,
    generation_config: GenerationConfig,
):
    # print("Model device:", next(model.parameters()).device)
    # print("Queries max token ID:", queries.max().item())
    # print("Queries min token ID:", queries.min().item())
    query_responses = []
    logitss = []
    batch_size = queries.shape[0]
    for i in range(0, batch_size, local_rollout_forward_batch_size):
        query = queries[i : i + local_rollout_forward_batch_size]
        query_response, logits = generate(
            model,
            query,
            pad_token_id,
            generation_config,
        )
        # print("query response, maybe summary", query_response)
        query_responses.append(query_response) 
        # TODO: made changes only returning responses
        # context_length = query.shape[1] 
        # response = query_response[:, context_length:]
        # query_responses.append(response)
        logitss.append(logits)

    # padding tensors
    padded_query_responses = pad(query_responses, padding_value=pad_token_id, padding_side="left")
    padded_logitss = pad(logitss, padding_value=0, padding_side="left")

    # reshaping
    padded_query_responses = padded_query_responses.view(-1, padded_query_responses.shape[-1])[:batch_size]
    padded_logitss = padded_logitss.view(-1, *padded_logitss.shape[2:])[:batch_size]

    return padded_query_responses, padded_logitss

# def batch_generation(
#     model,
#     queries,
#     batch_size,
#     pad_token_id,
#     generation_config,
# ):
#     """Generate responses for batched queries with improved error handling."""
    
#     # Ensure all inputs are on the same device
#     device = next(model.parameters()).device
#     queries = queries.to(device)
    
#     # Create attention mask
#     attention_mask = (queries != pad_token_id).to(device)
    
#     # Handle potential NaN/inf in the model
#     for param in model.parameters():
#         if torch.isnan(param.data).any() or torch.isinf(param.data).any():
#             param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1e4, neginf=-1e4)
    
#     try:
#         with torch.no_grad():
#             outputs = model.generate(
#                 input_ids=queries,
#                 attention_mask=attention_mask,
#                 generation_config=generation_config,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#             )
            
#         # Stack scores if available
#         if hasattr(outputs, 'scores') and outputs.scores:
#             logits = torch.stack(outputs.scores, dim=1)
#         else:
#             # Create dummy logits if scores not available
#             logits = torch.zeros(
#                 (queries.shape[0], 1, model.config.vocab_size),
#                 device=device
#             )
        
#         return outputs.sequences, logits
        
#     except RuntimeError as e:
#         if "probability tensor" in str(e):
#             print("Attempting generation with modified parameters...")
#             # Try with more conservative parameters
#             modified_config = GenerationConfig(
#                 max_new_tokens=10,
#                 do_sample=True,
#                 temperature=1.5,
#                 top_k=20,
#                 top_p=0.9,
#                 pad_token_id=pad_token_id,
#                 renormalize_logits=True
#             )
#             print("queries", queries)
#             outputs = model.generate(
#                 input_ids=queries,
#                 attention_mask=attention_mask,
#                 generation_config=modified_config,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#             )
            
#             logits = torch.stack(outputs.scores, dim=1)
#             return outputs.sequences, logits
#         else:
#             raise e

def add_bos_token_if_needed(
    bos_token_id: Optional[int],
    prompt_len_input_ids: int,
    prompt_tokens: Dict[str, List[int]],
    chosen_prompt_len_input_ids: int,
    chosen_tokens: Dict[str, List[int]],
    rejected_prompt_len_input_ids: int,
    rejected_tokens: Dict[str, List[int]],
):
    if bos_token_id is not None:
        if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
            prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
        if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
            chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
        if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
            rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]
    return prompt_tokens, chosen_tokens, rejected_tokens


def add_eos_token_if_needed(
    eos_token_id: int, chosen_tokens: Dict[str, List[int]], rejected_tokens: Dict[str, List[int]]
):
    if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
        chosen_tokens["input_ids"].append(eos_token_id)
        chosen_tokens["attention_mask"].append(1)
    if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
        rejected_tokens["input_ids"].append(eos_token_id)
        rejected_tokens["attention_mask"].append(1)
    return chosen_tokens, rejected_tokens


def truncate_right(
    input_ids: torch.Tensor, stop_token_id: int, pad_token_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Truncates the input tensor from the right side after the first occurrence of the stop token.

    Args:
        input_ids (`torch.Tensor`):
            The tensor containing the responses to be truncated
        stop_token_id (`int`):
            The token ID representing the stop token where truncation occurs
        pad_token_id (`int`):
            The token ID representing the pad token used to fill the truncated responses

    Returns:
        tuple:
            - `output_ids` (`torch.Tensor`):
                The truncated responses tensor with pad tokens filled after the stop token
            - `mask` (`torch.Tensor`):
                The mask tensor to indicate the padding tokens
    """
    trunc_idxs = first_true_indices(input_ids == stop_token_id).unsqueeze(-1)
    new_size = [1] * (len(input_ids.size()) - 1) + [input_ids.shape[1]]
    idxs = torch.arange(input_ids.shape[1], device=device).view(*new_size)
    output_ids = torch.masked_fill(input_ids, idxs > trunc_idxs, pad_token_id)
    mask = torch.masked_fill(torch.ones_like(input_ids), idxs > trunc_idxs, 0)
    return output_ids, mask


def empty_cache() -> None:
    """Empties the cache of the available torch device.

    This function checks for the availability of different torch devices (XPU, MLU, NPU, CUDA)
    and empties the cache of the first available device it finds.

    If none of the specific devices are available, it defaults to emptying the CUDA cache.
    """
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_mlu_available():
        torch.mlu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()


def decode_and_strip_padding(inputs: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> List[str]:
    """
    Decodes the input tensor and strips the padding tokens.

    Args:
        inputs (`torch.Tensor`):
            The input tensor to be decoded.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer used to decode the input tensor.

    Returns:
        `List[str]`:
            The list of decoded strings with padding tokens stripped.
    """
    decoded = tokenizer.batch_decode(inputs, skip_special_tokens=False)
    return [d.replace(tokenizer.pad_token, "") for d in decoded]

def generate_model_card(
    base_model: Optional[str],
    model_name: str,
    hub_model_id: str,
    dataset_name: Optional[str],
    tags: List[str],
    wandb_url: Optional[str],
    trainer_name: str,
    trainer_citation: Optional[str] = None,
    paper_title: Optional[str] = None,
    paper_id: Optional[str] = None,
) -> ModelCard:
    """
    Generate a `ModelCard` from a template.

    Args:
        base_model (`str` or `None`):
            Base model name.
        model_name (`str`):
            Model name.
        hub_model_id (`str`):
            Hub model ID as `username/model_id`.
        dataset_name (`str` or `None`):
            Dataset name.
        tags (`List[str]`):
            Tags.
        wandb_url (`str` or `None`):
            Weights & Biases run URL.
        trainer_name (`str`):
            Trainer name.
        trainer_citation (`str` or `None`, defaults to `None`):
            Trainer citation as a BibTeX entry.
        paper_title (`str` or `None`, defaults to `None`):
            Paper title.
        paper_id (`str` or `None`, defaults to `None`):
            ArXiv paper ID as `YYMM.NNNNN`.

    Returns:
        `ModelCard`:
            A ModelCard object.
    """
    card_data = ModelCardData(
        base_model=base_model,
        datasets=dataset_name,
        library_name="transformers",
        licence="license",
        model_name=model_name,
        tags=["generated_from_trainer", *tags],
    )
    card = ModelCard.from_template(
        card_data,
        template_path=str(pkg_resources.files("trl").joinpath("templates/lm_model_card.md")),
        base_model=base_model,
        model_name=model_name,
        hub_model_id=hub_model_id,
        dataset_name=dataset_name,
        wandb_url=wandb_url,
        trainer_name=trainer_name,
        trainer_citation=trainer_citation,
        paper_title=paper_title,
        paper_id=paper_id,
        trl_version=version("trl"),
        transformers_version=version("transformers"),
        pytorch_version=version("torch"),
        datasets_version=version("datasets"),
        tokenizers_version=version("tokenizers"),
    )
    return card