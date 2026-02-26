source activate ckd
export SEED=3467

accelerate launch --main_process_port 29500 lpd_reward.py \
  --output_dir /scratch4/mdredze1/jhuan236/logs/long_PDD/sd1/reward/1_component_T256_ELLA_init-768-8-6_unet_5e-5_$SEED \
  --ckpt /scratch4/mdredze1/jhuan236/logs/long_PDD/sd1/full/1_component_T256_ELLA_init-768-8-6_D10_1017/checkpoint-8000 \
  --num_components 1 \
  --num_tokens 256 \
  --decomposer_width 768 \
  --decomposer_heads 8 \
  --decomposer_layers 6 \
  --regu_weight 0.0 \
  --component_dropout 0.0 \
  --learning_rate 5.0e-5 \
  --mixed_precision bf16 \
  --tune_unet \
  --train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --checkpointing_steps 100 \
  --train_data_dir JourneyDB,coco,llava,sam \
  --max_train_steps 3000 \
  --validation_steps 50 \
  --gradient_checkpointing \
  --l2_norm_coeff 0.0 \
  --seed=$SEED \
  --resume_from_checkpoint latest \
  --pretrained_model_name_or_path /scratch4/mdredze1/jhuan236/zoo/runwayml/stable-diffusion-v1-5
