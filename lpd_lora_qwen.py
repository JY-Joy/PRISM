#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import contextlib
import copy
import functools
import gc
import logging
import math
import os
import random
import shutil

# Add repo root to path to import from tests
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.utils.checkpoint
from safetensors.torch import save_file, load_file
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed, DataLoaderConfiguration
from datasets import load_dataset, interleave_datasets
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig, TaskType

import diffusers
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageTransformer2DModel,
    QwenImagePipeline
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from pipeline_lpd_qwen import PromptDecomposePipeline
from models import PromptResampler, PromptResampler_ct5, PromptDecomposer, PromptDecomposer_v2
from tools import caption2embed, caption2embed_flux


if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def log_validation(decomposer, transformer, vae, text_encoder, tokenizer, args, accelerator, weight_dtype, epoch, comp_tokens):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = PromptDecomposePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(3467)
    images = []
    compose_image = []

    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask = encode_prompt(
            repeat_prompts_with_components(
                [args.validation_prompt],
                args.num_components,
                comp_tokens,
            ),
            tokenizer,
            text_encoder,
            special_tokens = comp_tokens,
            device = accelerator.device,
            dtype = weight_dtype,
        )
        neg_prompt_embeds, neg_prompt_embeds_mask = encode_prompt(
            repeat_prompts_with_components(
                [""],
                args.num_components,
                comp_tokens,
            ),
            tokenizer,
            text_encoder,
            special_tokens = comp_tokens,
            device = accelerator.device,
            dtype = weight_dtype,
        )
        image = pipeline(
            accelerator.unwrap_model(decomposer),
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=neg_prompt_embeds,
            negative_prompt_embeds_mask=neg_prompt_embeds_mask,
            # prompt="",
            num_inference_steps=20,
            height=512,
            width=512,
            generator=generator
        ).images[0]
        compose_image.append(image)
        image = pipeline.decompose(
            accelerator.unwrap_model(decomposer),
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=neg_prompt_embeds,
            negative_prompt_embeds_mask=neg_prompt_embeds_mask,
            # prompt="",
            height=512,
            width=512,
            num_inference_steps=20,
            generator=generator
        ).images
        images.extend(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "compose": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(compose_image)
                    ],
                    "decompose": [
                        wandb.Image(image, caption=f"Component {i}") for i, image in enumerate(images)
                    ],
                    # "masks": [
                    #     wandb.Image(
                    #         image.unsqueeze(-1).cpu().numpy(),
                    #         caption = f"mask {i}"
                    #     ) for i, image in enumerate(token_mask.detach())
                    # ],
                }
            )

    del pipeline
    free_memory()
    return images


class TextDecomposer(torch.nn.Module):
    def __init__(
        self,
        num_components=4,
        num_tokens=512,
        width=None,
        heads=None,
        layers=None,
        text_hidden_dim=None,
        unet_hidden_dim=None,
    ):
        super().__init__()
        self.num_components = num_components

        # self.mask_head = PromptResampler(
        #     width=width,
        #     heads=heads,
        #     layers=layers,
        #     num_tokens=num_tokens,
        #     num_components=num_components,
        #     input_dim=text_hidden_dim,
        #     output_dim=unet_hidden_dim,
        # )

        self.mask_head = torch.nn.Identity()
    
    def forward(self, encoder_hidden_state_t5,):
        bs, seq_len, hidden_dim = encoder_hidden_state_t5.shape
        encoder_hidden_state_list = self.mask_head(encoder_hidden_state_t5,)
        encoder_hidden_state_list = encoder_hidden_state_list.chunk(self.num_components, dim=0)
        return encoder_hidden_state_list


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--num_components",
        type=int,
        default=4,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=64,
        help="How many learnable tokens in resampler.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument("--token_length", type=int, default=512)
    parser.add_argument(
        "--decomposer_width",
        type=int,
        default=1024,
        help="Decomposer hidden dimension.",
    )
    parser.add_argument(
        "--decomposer_heads",
        type=int,
        default=8,
        help="The rank of the LoRA projection matrix.",
    )
    parser.add_argument(
        "--decomposer_layers",
        type=int,
        default=6,
        help="The rank of the LoRA projection matrix.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether or not to use center crop other wise random.",
    )
    parser.add_argument(
        "--component_dropout",
        type=float,
        default=0.1,
        help="The dropout probability for the dropout layer added before applying the LoRA to each layer input.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--upcast_vae",
        action="store_true",
        help="Whether or not to upcast vae to fp32",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help='previous ckpt',
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="A high angle shot of a brown wooden bench with several dishes on top of it. In the center and on the left are two round, wavy side plates with black scratches on the sides and a doily pattern engraved on the plates. On both plates is a thick brown cookie that's been crosscut at the top, located in the middle part of the image. The plate on the right has a candy with a yellow wrapper and green ends. To the right of the plates is a white mug with whipped cream on top that is similar to the glass plates. The cup, made of ceramic material, has a cylindrical shape with a handle and a textured surface. The white whipped cream on top is frothy and has an embossed design. Surrounding the wooden bench is a dark brown wooden floor. On the top right is a gray curtain, and on the upper left is a view of the lower part of a white wooden wall. The image is taken indoors with soft, warm lighting, likely from an overhead source, creating a cozy and inviting atmosphere. The lighting is evenly distributed, with no harsh shadows, suggesting a relaxed time of day, possibly evening. The style of the image is a realistic photo with a warm, homely aesthetic. The brown wooden bench supports the two round, wavy side plates with black scratches and a doily pattern, which are placed side by side. The thick brown cookies crosscut at the top are positioned on top of the two round, wavy side plates, with one cookie on each plate. The candy with a yellow wrapper and green ends is located on the right plate, next to the thick brown cookie. The white mug with whipped cream on top is situated to the right of the two round, wavy side plates. The two round, wavy side plates are adjacent to each other, with the plate containing the candy being closer to the white mug with whipped cream on top.",
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args

def repeat_prompts_with_components(prompts, num_components, special_tokens):

    formatted_prompts = []
    for prompt in prompts:
        
        prompt_parts = [
            f"{special_tokens[i]} {prompt}" for i in range(num_components)
        ]
        
        # 4. Join the parts with a space to create the final string
        formatted_string = " ".join(prompt_parts)
        formatted_prompts.append(formatted_string)
        
    return formatted_prompts

def encode_prompt(
    prompt,
    tokenizer,
    text_encoder,
    num_images_per_prompt = 1,
    prompt_embeds = None,
    prompt_embeds_mask = None,
    max_sequence_length = 512,
    special_tokens = None,
    device = None,
    dtype = None,
):
    r"""

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        device: (`torch.device`):
            torch device
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
    """

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]
    num_components = len(special_tokens)

    if prompt_embeds is None:
        prompt_embeds, prompt_embeds_mask = _get_qwen_prompt_embeds(prompt, tokenizer, text_encoder, special_tokens, device, dtype)
        # prompt_embeds, prompt_embeds_mask = get_qwen_prompt_embeds(prompt, tokenizer, text_encoder, special_tokens, device, dtype)

    prompt_embeds = prompt_embeds[:, :max_sequence_length]
    prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

    # _, seq_len, _ = prompt_embeds.shape
    # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    # prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt * num_components, seq_len, -1)
    # prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
    # prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

    return prompt_embeds, prompt_embeds_mask


def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
    bool_mask = mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

    return split_result

def _get_qwen_prompt_embeds(
    prompt,
    tokenizer,
    text_encoder,
    special_tokens: list,  # <-- NEW ARGUMENT
    device=None,
    dtype=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    # (Template setup remains the same)
    prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    prompt_template_encode_start_idx = 34
    tokenizer_max_length = 1024
    num_components = len(special_tokens)
    batch_size = len(prompt)

    template = prompt_template_encode
    drop_idx = prompt_template_encode_start_idx
    txt = [template.format(e) for e in prompt]

    txt_tokens = tokenizer(
        txt, max_length=tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    encoder_hidden_states = text_encoder(
        input_ids=txt_tokens.input_ids,
        attention_mask=txt_tokens.attention_mask,
        output_hidden_states=True,
    )
    hidden_states = encoder_hidden_states.hidden_states[-1]

    # --- 1. Process hidden_states AND input_ids in parallel ---
    trim_hidden_states = _extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
    trim_input_ids = _extract_masked_hidden(txt_tokens.input_ids, txt_tokens.attention_mask)

    # --- 2. Slice to remove template ---
    trim_hidden_states = [e[drop_idx:] for e in trim_hidden_states]
    trim_input_ids = [e[drop_idx:] for e in trim_input_ids]
    true_lengths = torch.tensor([e.size(0) for e in trim_input_ids], device=device)

    # --- 3. Find the indices of the special tokens ---
    component_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    component_token_ids.append(-1)  # handle last

    split_hidden_states, split_input_ids = [], []
    for start_id, end_id in zip(component_token_ids[:-1], component_token_ids[1:]):
        for hs, ids in zip(trim_hidden_states, trim_input_ids):
            start_loc = (ids == start_id).nonzero()
            if end_id >= 0:
                end_loc = (ids == end_id).nonzero()
            else:
                end_loc = torch.tensor([hs.size(0)], device=device)
            split_hidden_states.append(hs[start_loc:end_loc,...])
            split_input_ids.append(ids[start_loc:end_loc,...])

    # --- 4. pad to same length ---
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max([e.size(0) for e in split_hidden_states])

    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )    

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return prompt_embeds, encoder_attention_mask

# import torch.nn.utils.rnn as rnn_utils
def get_qwen_prompt_embeds(
    prompt,
    tokenizer,
    text_encoder,
    special_tokens: list,
    device=None,
    dtype=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    # (Template setup remains the same)
    prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    prompt_template_encode_start_idx = 34
    tokenizer_max_length = 1024
    num_components = len(special_tokens)
    batch_size = len(prompt)

    template = prompt_template_encode
    drop_idx = prompt_template_encode_start_idx
    txt = [template.format(e) for e in prompt]

    txt_tokens = tokenizer(
        txt, max_length=tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    encoder_hidden_states = text_encoder(
        input_ids=txt_tokens.input_ids,
        attention_mask=txt_tokens.attention_mask,
        output_hidden_states=True,
    )
    hidden_states = encoder_hidden_states.hidden_states[-1]

    # --- 1. Process hidden_states AND input_ids in parallel ---
    # This part is unchanged, as we get a list of variable-length tensors
    trim_hidden_states = _extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
    trim_input_ids = _extract_masked_hidden(txt_tokens.input_ids, txt_tokens.attention_mask)

    # --- 2. Slice to remove template ---
    # This list comprehension is necessary to handle the variable-length list
    trim_hidden_states = [e[drop_idx:] for e in trim_hidden_states]
    trim_input_ids = [e[drop_idx:] for e in trim_input_ids]
    true_lengths = torch.tensor([e.size(0) for e in trim_input_ids], device=device)

    # --- 3. Pad trimmed lists to create a batch ---
    # (Replaces loops from old Section 3 and 4)
    # We pad the list of tensors into a single batched tensor
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded_hidden_states = rnn_utils.pad_sequence(
        trim_hidden_states, batch_first=True, padding_value=0.0
    )
    padded_input_ids = rnn_utils.pad_sequence(
        trim_input_ids, batch_first=True, padding_value=pad_token_id
    )
    B, max_true_len, D = padded_hidden_states.shape

    # --- 4. Find component start/end indices in a batch ---
    component_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    start_ids_tensor = torch.tensor(component_token_ids, device=device).view(1, 1, -1)

    # Find where input_ids match the special token ids
    # Shape: [B, max_true_len, num_components]
    start_locs_mask = padded_input_ids.unsqueeze(-1) == start_ids_tensor
    
    # Use argmax to find the *first* occurrence (index) of each special token
    # Shape: [B, num_components]
    start_locs = start_locs_mask.int().argmax(dim=1)

    # End location of component `i` is the start location of component `i+1`
    end_locs = torch.roll(start_locs, shifts=-1, dims=1)
    
    # The last component's end_loc is the end of the sequence (true_lengths)
    end_locs[:, -1] = true_lengths

    # --- 5. Calculate component lengths and max length ---
    component_lengths = (end_locs - start_locs).clamp(min=0) # Shape: [B, C]
    max_seq_len = component_lengths.max() # This is the new max_seq_len
    
    # --- 6. Create indices for batched gathering ---
    # `relative_indices` goes from 0 to max_seq_len
    relative_indices = torch.arange(max_seq_len, device=device).view(1, 1, -1)
    # `start_locs_expanded` is the [B, C, 1] tensor of start indices
    start_locs_expanded = start_locs.unsqueeze(2)
    # `absolute_indices` maps relative indices to original sequence indices
    # Shape: [B, C, max_seq_len]
    absolute_indices = (relative_indices + start_locs_expanded)

    # Clamp indices to be valid (within 0 and max_true_len - 1)
    absolute_indices = torch.clamp(absolute_indices, 0, max_true_len - 1)

    # Expand indices to match the hidden dim (D) for gathering
    # Shape: [B, C, max_seq_len, D]
    indices_for_gather = absolute_indices.unsqueeze(-1).expand(-1, -1, -1, D)

    # --- 7. Gather hidden states ---
    # Expand hidden states to be gathered from: [B, 1, max_true_len, D] -> [B, C, max_true_len, D]
    hidden_states_expanded = padded_hidden_states.unsqueeze(1).expand(-1, num_components, -1, -1)
    
    # Perform the batched slice. This is the core replacement.
    # Shape: [B, C, max_seq_len, D]
    prompt_embeds = torch.gather(hidden_states_expanded, dim=2, index=indices_for_gather)

    # --- 8. Create new mask and apply it ---
    # The mask is True where relative index < component length
    # Shape: [B, C, max_seq_len]
    encoder_attention_mask = (relative_indices < component_lengths.unsqueeze(2))
    
    # Apply mask to zero out padding
    prompt_embeds = prompt_embeds * encoder_attention_mask.unsqueeze(-1)

    # --- 9. Reshape to final format ---
    # Reshape from [B, C, S, D] to [B*C, S, D]
    prompt_embeds = prompt_embeds.view(batch_size * num_components, max_seq_len, D)
    # Reshape from [B, C, S] to [B*C, S]
    encoder_attention_mask = encoder_attention_mask.view(batch_size * num_components, max_seq_len)

    # --- 10. Final conversion ---
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    # Convert boolean mask to long, as in the original
    encoder_attention_mask = encoder_attention_mask.long() 

    return prompt_embeds, encoder_attention_mask


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)     # optional: split_batches=True
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        dataloader_config=dataloader_config,
        kwargs_handlers=[ddp_kwargs]
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load the tokenizers
    tokenizer = Qwen2Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision, shift=3.0
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    latents_mean = (torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)).to(accelerator.device)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(accelerator.device)

    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        quantization_config=None,
        torch_dtype=weight_dtype,
    )
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, torch_dtype=weight_dtype
    )
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    decomposer = TextDecomposer(
        width=args.decomposer_width,
        heads=args.decomposer_heads,
        layers=args.decomposer_layers,
        num_tokens=args.num_tokens,
        num_components=args.num_components,
    )
    if args.ckpt is not None:
        test_sd = load_file(f"{args.ckpt}/model.safetensors", device="cpu")
        print(f'loading ckpt {args.ckpt}')
        decomposer.load_state_dict(test_sd)

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    decomposer.train()

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    comp_tokens = [f"<|component_{i}|>" for i in range(args.num_components)]
    special_token = {"additional_special_tokens": comp_tokens}
    # Add to tokenizer
    num_added_tokens = tokenizer.add_special_tokens(special_token)
    assert num_added_tokens == args.num_components
    new_token_ids = list(range(len(tokenizer) - args.num_components, len(tokenizer)))
    print(f"New token IDs: {new_token_ids}")

    # add lora on text-encoders
    qwen_lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,  # LoRA rank (a key hyperparameter)
        lora_alpha=64,  # LoRA alpha (scales the weights)
        lora_dropout=0.1,  # Dropout for LoRA layers
        bias="none",  # Set bias to "none" for LoRA
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
    )
    text_encoder = get_peft_model(text_encoder, qwen_lora_config)
    embed_layer = text_encoder.get_input_embeddings()
    org_emb = embed_layer.weight.data.clone()
    embed_layer.requires_grad_(True)
    text_encoder.train()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    lora_params = []
    embed_params = []

    for name, param in text_encoder.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora' in name:
            lora_params.append(param)
        elif 'embed_tokens' in name:
            # This will catch the embedding layer
            embed_params.append(param)
        else:
            # Catches other trainable params (like layer norms if specified)
            lora_params.append(param)

    lora_params += list(decomposer.parameters())
    # params_to_clip = list(decomposer.parameters()) + [p for n, p in text_encoder.named_parameters() if p.requires_grad]
    optimizer = optimizer_class(
        [
            {'params': lora_params, 'lr': args.learning_rate, 'betas': (args.adam_beta1, args.adam_beta2), 'weight_decay': args.adam_weight_decay, 'eps': args.adam_epsilon},
            {'params': embed_params, 'lr': args.learning_rate, 'betas': (args.adam_beta1, args.adam_beta2), 'weight_decay': 0.0, 'eps': args.adam_epsilon} # No decay on embeddings
        ],
    )

    def zero_grad_for_old_tokens(grad):
        # grad shape is [152064, 3584]
        # We want to keep grads for new_token_ids, zero out the rest
        mask = torch.zeros_like(grad)
        mask[new_token_ids, :] = 1.0  # Keep grads for new tokens
        return grad * mask

    # Register the hook on the embedding layer's weight
    embed_layer.weight.register_hook(zero_grad_for_old_tokens)

    # Move vae, transformer and text_encoder to device and cast to weight_dtype
    decomposer.to(accelerator.device)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    if "JourneyDB" in args.train_data_dir:
        # base_url = "https://huggingface.co/datasets/JourneyDB/JourneyDB/resolve/main/data/train/imgs/{i:03d}.tgz"
        # urls = [base_url.format(i=i) for i in range(200)]
        # train_dataset_jdb = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)
        train_dataset_jdb = load_dataset(
            '/scratch4/mdredze1/jhuan236/data/JourneyDB',
            cache_dir="/scratch4/mdredze1/jhuan236/data/JourneyDB/.cache",
            split="train",
            trust_remote_code=True,
            streaming=True,
        ).shuffle(seed=args.seed, buffer_size=10)
    if "coco" in args.train_data_dir:
        # train_dataset_coco = load_dataset(
        #     "webdataset",
        #     data_files={"train": '/scratch4/mdredze1/jhuan236/data/coco2017/train2017.zip'},
        #     split="train"
        # ).to_iterable_dataset(num_shards=200)
        train_dataset_coco = load_dataset(
            '/scratch4/mdredze1/jhuan236/data/coco2017',
            cache_dir=f"/scratch4/mdredze1/jhuan236/data/coco2017/.cache",
            split="train",
            trust_remote_code=True,
        ).to_iterable_dataset(num_shards=200).shuffle(seed=args.seed, buffer_size=10)
    if "llava" in args.train_data_dir:
        # train_dataset_lcs = load_dataset(
        #     "webdataset",
        #     data_files={"train": '/scratch4/mdredze1/jhuan236/data/LLaVA-Pretrain/images.zip'},
        #     split="train"
        # ).to_iterable_dataset(num_shards=200)
        train_dataset_lcs = load_dataset(
            '/scratch4/mdredze1/jhuan236/data/LLaVA-Pretrain',
            cache_dir=f"/scratch4/mdredze1/jhuan236/data/LLaVA-Pretrain/.cache",
            split="train",
            trust_remote_code=True,
        ).to_iterable_dataset(num_shards=200).shuffle(seed=args.seed, buffer_size=10)
    if "sam" in args.train_data_dir:
        # train_dataset_sam = load_dataset(
        #     '/scratch4/mdredze1/jhuan236/data/sa-1b',
        #     split="train",
        #     trust_remote_code=True,
        #     streaming=True,
        # ).shuffle(seed=args.seed, buffer_size=10)
        train_dataset_sam = load_dataset(
            '/scratch4/mdredze1/jhuan236/data/sa-1b',
            cache_dir=f"/scratch4/mdredze1/jhuan236/data/sa-1b/.cache",
            split="train",
            trust_remote_code=True,
        ).to_iterable_dataset(num_shards=200).shuffle(seed=args.seed, buffer_size=10)

    # preprocess
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    def preprocess_train(examples):
        if len(examples['image_path']) > 0:
            examples['pixel_values'] = Image.open(examples['image_path'])
        examples["pixel_values"] = train_transforms(
            examples['pixel_values'].convert("RGB")
        )
        return examples

    # probs = [0.3, 0.2, 0.25, 0.25]
    probs = [0.4, 0.10, 0.25, 0.25]
    # probs = [0.5, 0.05, 0.20, 0.25]
    # probs = [0.6, 0.05, 0.15, 0.20]
    train_dataset = interleave_datasets(
        [train_dataset_jdb, train_dataset_coco, train_dataset_lcs, train_dataset_sam],
        probabilities=probs,
        seed=args.seed,
        stopping_strategy='all_exhausted',
    )
    train_dataset = train_dataset.map(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        captions = [example['captions'] if random.random() >= args.component_dropout else '' for example in examples]
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(2)
        return {
            "pixel_values": pixel_values,
            "captions": captions,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = 20000
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    decomposer, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        decomposer, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(model, TextDecomposer):
                    save_file(model.state_dict(), os.path.join(output_dir, "model.safetensors"))
                else:
                    unet_sd = model.state_dict()
                    lora_sd = dict()
                    emb_sd = dict()
                    for n, p in unet_sd.items():
                        if 'lora' in n:
                            lora_sd[n] = p
                        elif 'embed_tokens' in n:
                            emb_sd[n] = p
                    save_file(lora_sd, os.path.join(output_dir, "text_encoder_lora.safetensors"))
                    save_file(emb_sd, os.path.join(output_dir, "embedding.safetensors"))
                weights.pop()

    def load_model_hook(models, input_dir):

        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, TextDecomposer):
                model.load_state_dict(load_file(os.path.join(input_dir, "model.safetensors"), device="cpu"))
            else:
                model.load_state_dict(load_file(os.path.join(input_dir, "text_encoder_lora.safetensors"), device="cpu"), strict=False)
                model.load_state_dict(load_file(os.path.join(input_dir, "embedding.safetensors"), device="cpu"), strict=False)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        accelerator.init_trackers("lpd_sd3", config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    trainable_module = (decomposer, text_encoder)
    for epoch in range(first_epoch, args.num_train_epochs):
        train_dataset.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(*trainable_module):
                with torch.no_grad():
                    # Convert images to latent space
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    model_input = vae.encode(pixel_values).latent_dist.sample()

                    # VAE normalization
                    model_input = (model_input - latents_mean) * latents_std
                    model_input = model_input.to(dtype=weight_dtype)

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0]

                    # Sample a random timestep for each image
                    # for weighting schemes where we sample timesteps non-uniformly
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                    # Add noise according to flow matching. zt = (1 - texp) * x + texp * z1
                    sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                    # Predict the noise residual
                    img_shapes = [
                        (1, args.resolution // vae_scale_factor // 2, args.resolution // vae_scale_factor // 2)
                    ] * bsz
                    # transpose the dimensions
                    noisy_model_input = noisy_model_input.permute(0, 2, 1, 3, 4)
                    packed_noisy_model_input = QwenImagePipeline._pack_latents(
                        noisy_model_input,
                        batch_size=model_input.shape[0],
                        num_channels_latents=model_input.shape[1],
                        height=model_input.shape[3],
                        width=model_input.shape[4],
                    )

                # Get the text embedding for conditioning
                # component-first order
                prompt_embeds, prompt_embeds_mask = encode_prompt(
                    repeat_prompts_with_components(
                        batch['captions'],
                        args.num_components,
                        comp_tokens,
                    ),
                    tokenizer,
                    text_encoder,
                    special_tokens = comp_tokens,
                    device = accelerator.device,
                    dtype = weight_dtype,
                )
                token_hidden_states = decomposer(prompt_embeds)
                prompt_embeds = torch.cat(token_hidden_states).to(dtype=weight_dtype)       # component-first order

                # Predict the noise residual
                model_pred_batch = transformer(
                    hidden_states=torch.cat([packed_noisy_model_input]*args.num_components),
                    timestep=torch.cat([timesteps / 1000]*args.num_components),
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    img_shapes=img_shapes*args.num_components,
                    txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
                    return_dict=False,
                )[0]
                model_pred_batch = QwenImagePipeline._unpack_latents(
                    model_pred_batch, args.resolution, args.resolution, vae_scale_factor
                )

                model_pred = sum(model_pred_batch.chunk(args.num_components)) / args.num_components

                # flow matching loss
                target = noise - model_input

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                loss = 0.0
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # Compute regular loss.
                diff_loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                ).mean()
                loss += diff_loss

                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     params_to_clip = params_to_optimize
                #     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # # Freeze other embeddings
                # with torch.no_grad():
                #     unwrap_model(text_encoder).get_input_embeddings().weight[:151665] = (
                #         org_emb[:151665]
                #     )
                #     # unwrap_model(text_encoder).get_input_embeddings().weight[151667:] = (
                #     #     org_emb[151667:]
                #     # )
                #     embed_layer = unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
                #     assert (org_emb-embed_layer)[:151665].sum() <= 1e-6

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        images = log_validation(
                            decomposer, transformer, vae, unwrap_model(text_encoder), tokenizer, args, accelerator, weight_dtype, epoch, comp_tokens
                        )

            logs = {
                # "loss": kd_loss.detach().item(),
                "diff": diff_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)