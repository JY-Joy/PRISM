source activate ckd
export SEED=3467

accelerate launch --main_process_port 29500 lpd_ella.py \
  --output_dir /scratch4/mdredze1/jhuan236/logs/long_PDD/sd1/full/4_component_T64_ELLA_init_Regu-768-8-6_D10_$SEED \
  --ckpt ella \
  --num_components 4 \
  --num_tokens 64 \
  --decomposer_width 768 \
  --decomposer_heads 8 \
  --decomposer_layers 6 \
  --regu_weight 0.1 \
  --component_dropout 0.1 \
  --lr_scheduler constant_with_warmup \
  --learning_rate 5.0e-5 \
  --lr_warmup_steps 2000 \
  --mixed_precision bf16 \
  --train_batch_size  20 \
  --gradient_accumulation_steps 2 \
  --checkpointing_steps 2000 \
  --train_data_dir JourneyDB,coco,llava,sam \
  --max_train_steps 10000 \
  --validation_steps 500 \
  --l2_norm_coeff 0.0 \
  --seed $SEED \
  --train_size=512 \
  --resume_from_checkpoint latest \
  --pretrained_model_name_or_path /scratch4/mdredze1/jhuan236/zoo/runwayml/stable-diffusion-v1-5
  --ckpt /scratch4/mdredze1/jhuan236/logs/long_PDD/sd1/full/4_component_T64_ELLA_init-768-8-6_D10_3467/checkpoint-8000 \