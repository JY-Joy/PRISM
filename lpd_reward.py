#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

# system
import argparse
import logging
import math
import os
import re
import json
import random
import shutil
import warnings
import time

# utilities
import numpy as np
import PIL
import safetensors
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint
from contextlib import nullcontext
from accelerate import Accelerator
from safetensors.torch import save_file, load_file
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, DataLoaderConfiguration, set_seed
from tools import caption2embed, ids2embed, get_predicted_original, DDIMSolver

# data
from PIL import Image
from packaging import version
from torchvision import transforms
from datasets import load_dataset, interleave_datasets
from tqdm.auto import tqdm
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    AutoTokenizer,
    T5EncoderModel,
)
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

# models
import diffusers as diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.models.attention_processor import AttnProcessor
from diffusers.image_processor import IPAdapterMaskProcessor

from pipeline_prompt_decomposition import PromptDecomposePipeline
from models import MLP, PromptResampler
from loss import dense_score
from lpd_ella import TextDecomposer


if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

# ------------------------------------------------------------------------------

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.32.0.dev0")

logger = get_logger(__name__)


def get_activation(mem, name):
    def get_output_hook(module, input, output):
        mem[name] = output[1]

    return get_output_hook

def add_hook(net, mem, mapping_layers):
    for n, m in net.named_modules():
        if n in mapping_layers:
            m.register_forward_hook(get_activation(mem, n))


def get_module_kohya_state_dict(module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"):
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(module, adapter_name=adapter_name).items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f"{kohya_key.split('.')[0]}.alpha"
            kohya_ss_state_dict[alpha_key] = torch.tensor(module.peft_config[adapter_name].lora_alpha).to(dtype)

    return kohya_ss_state_dict


def log_validation(decomposer, caption2embed_simple, unet, vae, args, accelerator, weight_dtype, epoch, return_mask=False):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = PromptDecomposePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        # text_encoder=text_encoder,
        # tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    compose_image = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    captions = ["", args.validation_prompt]
    encoder_hidden_states_pre = caption2embed_simple(captions)
    encoder_hidden_states_clip = encoder_hidden_states_pre['encoder_hidden_states_clip_concat']
    encoder_hidden_states_t5 = encoder_hidden_states_pre["encoder_hidden_states_t5"]
    encoder_hidden_states = encoder_hidden_states_t5
    with autocast_ctx:
        image = pipeline(
            accelerator.unwrap_model(decomposer),
            # encoder_hidden_states_t5,
            # encoder_hidden_states_clip,
            prompt_embeds = encoder_hidden_states[1:],
            negative_prompt_embeds = encoder_hidden_states[:1],
            num_inference_steps=25,
            generator=generator
        ).images[0]
    compose_image.append(image)
    with autocast_ctx:
        image = pipeline.decompose(
            accelerator.unwrap_model(decomposer),
            # encoder_hidden_states_t5,
            # encoder_hidden_states_clip,
            prompt_embeds = encoder_hidden_states[1:],
            negative_prompt_embeds = encoder_hidden_states[:1],
            num_inference_steps=25,
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
    torch.cuda.empty_cache()
    return images


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
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
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument("--token_length", type=int, default=512)
    parser.add_argument(
        "--decomposer_width",
        type=int,
        default=1024,
        help="The rank of the LoRA projection matrix.",
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
        "--regu_weight",
        type=float,
        default=0.0,
        help="The dropout probability for the dropout layer added before applying the LoRA to each layer input.",
    )
    parser.add_argument(
        "--l2_norm_coeff",
        type=float,
        default=0.0,
        help="The dropout probability for the dropout layer added before applying the LoRA to each layer input.",
    )
    parser.add_argument(
        "--component_dropout",
        type=float,
        default=0.1,
        help="The dropout probability for the dropout layer added before applying the LoRA to each layer input.",
    )
    parser.add_argument(
        "--dropout_v1",
        action="store_true",
        help="Whether or not to use dropout v1: drop some components in a batch.",
    )
    parser.add_argument(
        "--tune_unet",
        action="store_true",
        help="Whether or not to tune unet LoRA.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="The rank of the LoRA projection matrix.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="The dropout probability for the dropout layer added before applying the LoRA to each layer input.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help=(
            "A comma-separated string of target module keys to add LoRA to. If not set, a default list of modules will"
            " be used. By default, LoRA will be applied to all conv and linear layers."
        ),
    )
    parser.add_argument(
        "--learn_temperature",
        action="store_true",
        help="Whether or not to learn tempreture in CWA.",
    )
    parser.add_argument(
        "--compositional_entropy",
        action="store_true",
        help=(
            "Whether or not to use compositional entropy regulariser"
        ),
    )
    parser.add_argument(
        "--cwa",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, help="A folder containing the training data."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_size",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="bilinear",
        help=(
            'train data processing interpolation mode'
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=6000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
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
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
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
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
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
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and Nvidia Ampere GPU or Intel Gen 4 Xeon (and later) ."
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
        "--validation_prompt",
        type=str,
        default='''
An overhead view of four labradoodle puppies, three puppies are sitting and one puppy is standing with its right paw resting against the white barrier at the bottom of the image. The puppies are on a light blue rug placed on a black floor. The puppy standing is a beige and white puppy with curly fur, dark eyes, a small nose, and a fluffy appearance, its paw extended. There is a black and white puppy sitting on its hind legs to the right, and to the left part of the image is another beige puppy sitting on its hind legs as well. Directly behind the standing puppy, in the upper part of the image, is another light cream colored puppy sitting on its hind legs, looking toward the bottom right corner of the image. The three puppies in the front are looking up, the puppy behind them is looking toward the bottom right corner of the image. There is a blue plush toy in the bottom right corner of the image underneath the black puppy. The rug the puppies are on is not laying completely flat on the ground, its unintentionally folded up in some areas and folded over itself in the top right corner of the image. The background consists of a light blue rug placed on a black floor, with the rug showing some unintentional folds and overlaps. A blue plush toy is visible in the bottom right corner under the black puppy. The image is well-lit with soft, even lighting, suggesting an indoor setting with artificial light sources. The light appears to be front-lit, as there are no harsh shadows on the puppies. The style of the image is a realistic photo. The beige and white puppy standing with its right paw resting against the white barrier is in front of the light cream colored puppy sitting on its hind legs in the back. The black and white puppy sitting on its hind legs to the right is to the right of the beige and white puppy standing with its right paw resting against the white barrier. The beige puppy sitting on its hind legs to the left is to the left of the beige and white puppy standing with its right paw resting against the white barrier. The light cream colored puppy sitting on its hind legs in the back is behind the beige and white puppy standing with its right paw resting against the white barrier. The black and white puppy sitting on its hind legs to the right is next to the beige puppy sitting on its hind legs to the left.''',
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help='previous ckpt',
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
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)     # optional: split_batches=True
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        dataloader_config=dataloader_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

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

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=weight_dtype)
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    args.num_ddim_timesteps = 50
    solver = DDIMSolver(noise_scheduler.alphas_cumprod.numpy(),
                        timesteps=noise_scheduler.config.num_train_timesteps,
                        ddim_timesteps=args.num_ddim_timesteps)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    tokenizer_t5 = AutoTokenizer.from_pretrained("/scratch4/mdredze1/jhuan236/zoo/ella/models--google--flan-t5-xl--text_encoder")
    text_encoder_t5 = T5EncoderModel.from_pretrained("/scratch4/mdredze1/jhuan236/zoo/ella/models--google--flan-t5-xl--text_encoder")
    decomposer = TextDecomposer(
        width=args.decomposer_width,
        heads=args.decomposer_heads,
        layers=args.decomposer_layers,
        num_tokens=args.num_tokens,
        num_components=args.num_components,
        text_hidden_dim=text_encoder_t5.config.d_model,
    )
    if args.ckpt is not None:
        test_sd = load_file(f"{args.ckpt}/model.safetensors", device="cpu")
        print(f'loading ckpt {args.ckpt}')
        decomposer.load_state_dict(test_sd)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    # Freeze vae and unet
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_t5.requires_grad_(False)

    # Move vae and unet to device and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    decomposer.to(accelerator.device)
    text_encoder_t5.to(accelerator.device, weight_dtype)

    alpha_schedule = alpha_schedule.to(accelerator.device, weight_dtype)
    sigma_schedule = sigma_schedule.to(accelerator.device, weight_dtype)
    get_predicted_original_sample = lambda model_output, timesteps, sample: get_predicted_original(
        model_output, timesteps, sample, noise_scheduler.config.prediction_type, alpha_schedule, sigma_schedule)
    solver = solver.to(accelerator.device, weight_dtype)

    # Add LoRA to the U-Net, only the LoRA projection matrix will be updated by the optimizer.
    if args.tune_unet:
        decomposer.to(dtype=weight_dtype)
        decomposer.requires_grad_(False)
        if args.lora_target_modules is not None:
            lora_target_modules = [module_key.strip() for module_key in args.lora_target_modules.split(",")]
        else:
            lora_target_modules = [
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
                "conv1",
                "conv2",
                "conv_shortcut",
                "downsamplers.0.conv",
                "upsamplers.0.conv",
                "time_emb_proj",
            ]
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
        )
        unet.add_adapter(lora_config)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters into fp32
        cast_training_params(unet, dtype=torch.float32)

    # Component-wise Attention
    if args.cwa:
        attn_procs = {}
        for k,v in unet.attn_processors.items():
            if '.attn2.' in k:
                attn_procs[k] = DecomposingAttnProcessor_pool(
                    args.num_components,
                    learn_temperature=args.learn_temperature,
                    dtype=torch.float32 if weight_dtype==torch.float16 else weight_dtype
                )
            else:
                attn_procs[k] = AttnProcessor()
        unet.set_attn_processor(attn_procs)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            # teacher_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    params_to_optimize = [p for n, p in unet.named_parameters() if p.requires_grad] if args.tune_unet else list(decomposer.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    if args.debug:
        from datasets import Dataset
        def dummy_data_generator():
            for i in range(100):
                yield {
                    "res": args.train_size,
                    "caption": "",
                }
        train_dataset = Dataset.from_generator(dummy_data_generator)
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

    caption2embed_simple = lambda captions: caption2embed(captions, [tokenizer, tokenizer_t5], [text_encoder, text_encoder_t5],
                                                          accelerator.device, weight_dtype, args=args, token_length=args.token_length)

    # Preprocessing the datasets.
    args.interpolation = {
        "linear": PIL_INTERPOLATION["linear"],
        "bilinear": PIL_INTERPOLATION["bilinear"],
        "bicubic": PIL_INTERPOLATION["bicubic"],
        "lanczos": PIL_INTERPOLATION["lanczos"],
    }[args.interpolation]
    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.train_size) if args.center_crop else transforms.RandomCrop(args.train_size),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples: dict):
        results = dict()
        # image key is jpg
        results["pixel_values"] = train_transforms(
            examples['pixel_values'].convert("RGB").resize(
                (args.train_size, args.train_size), resample=args.interpolation
            )
        )

        # # captions
        # if examples['__key__'].endswith('.jpg'):
        #     # sam and journeydb
        #     if examples['__key__'].startswith('sa'):
        #         path = 'sam-570k/images/'+examples['__key__']
        #         if path not in longcaption['sam']: raise KeyError
        #         results['captions'] = longcaption['sam'][path]
        #     else:
        #         path = "journeydb/images/"+examples['__key__']
        #         if path not in longcaption['journeydb']: raise KeyError
        #         results['captions'] = longcaption['journeydb'][path]
        # else:
        #     # coco and lcs
        #     if 'train2017' in examples['__key__']:
        #         path = f"coco2017/{examples['__key__']}.jpg"
        #         if path not in longcaption['coco']: raise KeyError
        #         results['captions'] = longcaption['coco'][path]
        #     else:
        #         path = f"lcs-558k/images/{examples['__key__']}.jpg"
        #         if path not in longcaption['llava']: raise KeyError
        #         results['captions'] = longcaption['llava'][path]
        return results
  
    # probs = [0.4, 0.10, 0.25, 0.25]
    probs = [0.5, 0.05, 0.20, 0.25]
    # probs = [0.6, 0.05, 0.15, 0.20]

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def collate_fn(examples):
        # parse captions
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        results = {"pixel_values": pixel_values}
        captions = [example['captions'] if random.random() >= args.component_dropout else '' for example in examples]
        # tokens_t5 = tokenizer_t5(captions, max_length=tokenizer_t5.model_max_length,
        #                          padding="max_length", truncation=True, return_tensors="pt")
        # results["input_ids_t5"], results["attention_mask_t5"] = tokens_t5.input_ids, tokens_t5.attention_mask
        results['captions'] = captions
        return results

    # Get the text embedding for conditioning
    uncond_prompt_embeds_pre = caption2embed_simple([""]*args.train_batch_size)
    # uncond_hidden_states_clip = uncond_prompt_embeds_pre['encoder_hidden_states_clip_concat']
    uncond_hidden_states_t5 = uncond_prompt_embeds_pre["encoder_hidden_states_t5"]

    # DataLoaders creation:
    train_dataset = interleave_datasets(
        [train_dataset_jdb, train_dataset_coco, train_dataset_lcs, train_dataset_sam],
        probabilities=probs,
        seed=args.seed,
        stopping_strategy='all_exhausted',
    )
    train_dataset = train_dataset.map(preprocess_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    # Prepare everything with our `accelerator`.
    if args.tune_unet or args.learn_temperature:
        unet.train()
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
        trainable_module = (unet,)
    else:
        decomposer.train()
        decomposer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            decomposer, optimizer, train_dataloader, lr_scheduler
        )
        trainable_module = (decomposer,)

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(model, TextDecomposer):
                    save_file(model.state_dict(), os.path.join(output_dir, "model.safetensors"))
                else:
                    unet_sd = model.state_dict()
                    save_sd = dict()
                    for n, p in unet_sd.items():
                        if 'lora' in n or 'temperature' in n:
                            save_sd[n] = p
                    save_file(save_sd, os.path.join(output_dir, "unet.safetensors"))
                weights.pop()

    def load_model_hook(models, input_dir):

        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, TextDecomposer):
                model.load_state_dict(load_file(os.path.join(input_dir, "model.safetensors"), device="cpu"))
            else:
                model.load_state_dict(load_file(os.path.join(input_dir, "unet.safetensors"), device="cpu"), strict=False)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = args.max_train_steps
    if overrode_max_train_steps:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("long_prompt_decompose", config=vars(args))
    
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    # logger.info(f"  Total optimization steps = {args.max_train_samples}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = args.resume_from_checkpoint
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
            if path.startswith("checkpoint"):
                accelerator.load_state(os.path.join(args.output_dir, path))
                global_step = int(path.split("-")[1])
            else:
                accelerator.load_state(args.resume_from_checkpoint)
                global_step = 0

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

    # Add hook for attn map regu
    acts_stu = {}
    mapping_layers_stu = [
        'down_blocks.0.attentions.0.transformer_blocks.0.attn2', 'down_blocks.0.attentions.1.transformer_blocks.0.attn2',
        'down_blocks.1.attentions.0.transformer_blocks.0.attn2', 'down_blocks.1.attentions.1.transformer_blocks.0.attn2',
        'down_blocks.2.attentions.0.transformer_blocks.0.attn2', 'down_blocks.2.attentions.1.transformer_blocks.0.attn2',
        'mid_block.attentions.0.transformer_blocks.0.attn2',
        'up_blocks.1.attentions.0.transformer_blocks.0.attn2', 'up_blocks.1.attentions.1.transformer_blocks.0.attn2', 'up_blocks.1.attentions.2.transformer_blocks.0.attn2',
        'up_blocks.2.attentions.0.transformer_blocks.0.attn2', 'up_blocks.2.attentions.1.transformer_blocks.0.attn2', 'up_blocks.2.attentions.2.transformer_blocks.0.attn2',
        'up_blocks.3.attentions.0.transformer_blocks.0.attn2', 'up_blocks.3.attentions.1.transformer_blocks.0.attn2', 'up_blocks.3.attentions.2.transformer_blocks.0.attn2',
    ]
    if args.compositional_entropy:
        add_hook(unet, acts_stu, mapping_layers_stu)

    # Training loop
    # ortho_weight = 1. - float(os.environ.get('REWEIGHT', 0.3))
    ortho_weight = 1. - 0.3
    tic = time.time()
    for epoch in range(first_epoch, args.num_train_epochs):
        train_dataset.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(*trainable_module):
                with torch.no_grad():
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    if global_step % 2 == 1:
                        start_timesteps = torch.randint(noise_scheduler.config.num_train_timesteps // 2,
                                                        noise_scheduler.config.num_train_timesteps,
                                                        (latents.shape[0],), device=latents.device).long()
                    else:
                        start_timesteps = torch.randint(8, noise_scheduler.config.num_train_timesteps // 2,
                                                        (latents.shape[0],), device=latents.device).long()

                    # OPTIONAL: start from an intermediate step
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    timesteps = start_timesteps - solver.step_ratio
                    timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)

                    # Sample a random guidance scale w from U[w_min, w_max]
                    args.w_min = args.w_max = 7.5
                    w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                    w = w.reshape(bsz, 1, 1, 1)
                    w = w.to(device=latents.device, dtype=latents.dtype)

                    # Get the text embedding for conditioning
                    encoder_hidden_states_pre = caption2embed_simple(batch['captions'])
                    # encoder_hidden_states_clip = encoder_hidden_states_pre['encoder_hidden_states_clip_concat']
                    # encoder_hidden_states_clip_ = torch.cat([encoder_hidden_states_clip, uncond_hidden_states_clip], dim=0)
                    encoder_hidden_states_t5 = encoder_hidden_states_pre["encoder_hidden_states_t5"]
                    encoder_hidden_states_t5_ = torch.cat([encoder_hidden_states_t5, uncond_hidden_states_t5], dim=0)

                # Generating from noise with DDIM
                time_all = list(range(999, 0, -20))
                rand_start = random.randint(0, 9)
                time_bp = time_all[rand_start::10]
                x_prev = noise
                for t in time_all:
                    x_prev_ = x_prev.repeat(2*args.num_components, 1, 1, 1)
                    t_ = t * torch.ones_like(timesteps).repeat(2)
                    token_hidden_states = decomposer(encoder_hidden_states_t5_, t_)     # a list of length `num_components`, each element is a batch
                    # num_components x [batch | ug_batch]
                    decomposed_encoder_hidden_states = torch.cat(token_hidden_states).to(dtype=weight_dtype)
                    if t in time_bp:
                        noise_pred = checkpoint.checkpoint(
                            unet,
                            x_prev_.detach(),
                            torch.cat([t_]*args.num_components),
                            decomposed_encoder_hidden_states,
                            use_reentrant=False
                        ).sample
                    else:
                        with torch.no_grad():
                            noise_pred = unet(
                                x_prev_,
                                torch.cat([t_]*args.num_components),
                                decomposed_encoder_hidden_states
                            ).sample
                    noise_pred = noise_pred.reshape(args.num_components, 2, bsz, *noise_pred.shape[1:])
                    noise_pred = noise_pred[:,1] + w * (noise_pred[:,0] - noise_pred[:,1])
                    noise_pred = noise_pred.mean(dim=0)
                    pred_x_0 = get_predicted_original_sample(noise_pred, t * torch.ones_like(timesteps), x_prev)
                    x_prev = solver.ddim_step(pred_x_0, noise_pred, (t - 20) * torch.ones_like(timesteps),
                                              is_prev=True).to(weight_dtype)

                pred_x_0 = vae.decode(pred_x_0.to(weight_dtype) / vae.config.scaling_factor, return_dict=False)[0]
                pred_x_0 = (pred_x_0 * 0.5 + 0.5).clamp(0, 1)
                score = dense_score(pred_x_0, encoder_hidden_states_pre['caption_split'], encoder_hidden_states_pre['caption_index'], do_ortho=True, accelerator=accelerator)
                # score = dense_score_plus(pred_x_0, batch['caption_split'], batch['caption_index'], do_ortho=True, accelerator=accelerator)
                # loss = 0.5 - (torch.diagonal(score).mean() - score.mean() * 0.3)
                loss = (1 - score.mean()) * 0.5 - 0.09 * ortho_weight
                # loss = ((1 - dense_score(pred_x_0, batch['caption_split'], batch['caption_index'])) * 0.5).clamp(0.365).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, 0.3)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

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

                        if args.tune_unet:
                            unwrapped_unet = unwrap_model(unet)
                            unet_lora_state_dict = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(unwrapped_unet)
                            )

                            StableDiffusionPipeline.save_lora_weights(
                                save_directory=save_path,
                                unet_lora_layers=unet_lora_state_dict,
                                safe_serialization=True,
                            )

                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        images = log_validation(
                            decomposer, caption2embed_simple, unet, vae, args, accelerator, weight_dtype, epoch
                        )

            logs = {
                "loss": loss.detach().item(),
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
