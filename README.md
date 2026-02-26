# Long Prompt Decomposition (LPD)

PyTorch implementation for decomposing long text prompts in text-to-image generation models.

## Overview

This repository implements a Long Prompt Decomposition (LPD) method that enables diffusion models to handle long, complex text prompts by decomposing them into multiple semantic components. Each component is processed independently through a learned decomposer network, allowing the model to generate images that better capture all aspects of detailed prompts.

## Features

- **Multiple Model Support**: Compatible with SD1.5 (Stable Diffusion 1.5) and Qwen-Image models
- **Flexible Architecture**: Uses LoRA adapters for efficient fine-tuning
- **Component-Based Decomposition**: Decomposes prompts into learnable semantic components
- **Training Scripts**: Ready-to-use training scripts with accelerate for multi-GPU training
- **Inference Pipelines**: Custom diffusion pipelines for prompt decomposition

## Installation

```bash
git clone <repo-url>
cd prism
pip install -r requirements.txt
```

## Training

### SD1.5 (Stable Diffusion 1.5)

```bash
bash lpd_sd1.sh
```

### Qwen-Image

```bash
bash lpd_qwen.sh
```

### Reward Finetuning

```bash
bash lpd_reward.sh
```

## Inference

### SD1.5

#### Command-line Interface

```bash
# Single prompt
python sample_pipeline.py \
  --ckpt /path/to/checkpoint \
  --prompt "A detailed landscape painting of mountains..."

# Batch from file
python sample_pipeline.py \
  --ckpt /path/to/checkpoint \
  --prompt_file prompts.txt \
  --output_dir ./results

# With custom settings
python sample_pipeline.py \
  --ckpt /path/to/checkpoint \
  --prompt "Your prompt here" \
  --num_inference_steps 30 \
  --guidance_scale 8.0 \
  --height 768 \
  --width 768 \
  --seed 123
```

#### Python API

```python
from pipeline_prompt_decomposition import PromptDecomposePipeline
from models import PromptDecomposer

# Load pipeline
pipeline = PromptDecomposePipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Load decomposer checkpoint
decomposer = PromptDecomposer.from_pretrained("/path/to/checkpoint")

# Generate image
image = pipeline(
    decomposer=decomposer,
    prompt="A detailed description of a scene...",
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]
```

### Qwen-Image

```bash
python lpd_qwen_infer.py \
  --pretrained_model_name_or_path /path/to/Qwen-Image \
  --ckpt /path/to/checkpoint \
  --prompt "Your long prompt here..."
```

## Key Components

### Training Scripts
- **`lpd_lora_qwen.py`**: Training script for Qwen-Image model
- **`lpd_ella.py`**: Training script for SD1.5 model
- **`lpd_reward.py`**: Reward-based fine-tuning script

### Inference Scripts
- **`sample_pipeline.py`**: Command-line inference for SD1.5
- **`lpd_qwen_infer.py`**: Command-line inference for Qwen-Image

### Core Modules
- **`models.py`**: Core model architectures including `PromptDecomposer`, `PromptResampler`, and `MLP`
- **`tools.py`**: Utility functions for text encoding and embeddings
- **`loss.py`**: Loss functions for training
- **`pipeline_prompt_decomposition.py`**: Custom diffusion pipeline for SD1.5
- **`pipeline_lpd_qwen.py`**: Custom diffusion pipeline for Qwen-Image

## Citation

```bibtex
@article{your-paper,
  title={Long Prompt Decomposition for Text-to-Image Generation},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## License

Please refer to the LICENSE file for licensing information.
