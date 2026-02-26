source activate lpd
export SEED=3467

accelerate launch --main_process_port 29500 lpd_lora_qwen.py \
  --output_dir /scratch4/mdredze1/jhuan236/logs/long_PDD/qwen/2_component_lora_r32_D10_5e-5_$SEED \
  --num_components 2 \
  --num_tokens 512 \
  --decomposer_width 3584 \
  --decomposer_heads 64 \
  --decomposer_layers 1 \
  --component_dropout 0.1 \
  --lr_scheduler constant_with_warmup \
  --mixed_precision bf16 \
  --resolution 512 \
  --learning_rate 5e-5 \
  --gradient_checkpointing \
  --train_batch_size 6 \
  --gradient_accumulation_steps 1 \
  --checkpointing_steps 500 \
  --train_data_dir JourneyDB,coco,llava,sam \
  --max_train_steps 30000 \
  --validation_steps 500 \
  --resume_from_checkpoint latest \
  --upcast_vae \
  --seed $SEED \
  --pretrained_model_name_or_path /scratch4/mdredze1/jhuan236/zoo/Qwen/Qwen-Image
  --ckpt /scratch4/mdredze1/jhuan236/logs/long_PDD/sd3/4_component_t5_T128-4096-32-6_D10_1021/checkpoint-12000 \