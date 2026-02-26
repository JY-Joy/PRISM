#!/usr/bin/env python
"""
Inference script for Long Prompt Decomposition with SD1.5.
Simplified interface for easy command-line usage.
"""

import argparse
import os
import torch
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from pipeline_prompt_decomposition import PromptDecomposePipeline
from models import PromptDecomposer


def main():
    parser = argparse.ArgumentParser(description="LPD Inference for SD1.5")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="runwayml/stable-diffusion-v1-5",
                        help="Path to pretrained SD1.5 model")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to LPD checkpoint directory")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to generate")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="File containing prompts (one per line)")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale for CFG")
    parser.add_argument("--height", type=int, default=512,
                        help="Image height")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for generated images")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Data type for model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set device and dtype
    device = torch.device(args.device)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Set random seed
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading pipeline from {args.pretrained_model_name_or_path}...")
    # Load base pipeline
    base_pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        safety_checker=None,
    )

    # Create custom LPD pipeline
    pipeline = PromptDecomposePipeline(
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        tokenizer=base_pipe.tokenizer,
        unet=base_pipe.unet,
        scheduler=base_pipe.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(device)

    # Use DPM++ solver for better quality
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    print(f"Loading decomposer from {args.ckpt}...")
    # Load decomposer checkpoint
    decomposer = PromptDecomposer.from_pretrained(args.ckpt)
    decomposer = decomposer.to(device, dtype=dtype)
    decomposer.eval()

    # Get prompts
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default example prompts
        prompts = [
            "A detailed landscape painting of a mountain range at sunset, with a serene lake in the foreground reflecting the colorful sky, surrounded by pine trees and wildflowers",
            "A futuristic cityscape with towering skyscrapers, flying vehicles, neon lights, and holographic advertisements, set against a dark sky with multiple moons",
        ]
        print("No prompt provided, using example prompts")

    print(f"Generating {len(prompts)} image(s)...")

    # Generate images
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] Generating: {prompt[:100]}...")

        with torch.no_grad():
            output = pipeline(
                decomposer=decomposer,
                prompt=prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )

        # Save image
        image = output.images[0]
        save_path = os.path.join(args.output_dir, f"image_{idx:04d}.png")
        image.save(save_path)
        print(f"Saved to {save_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
