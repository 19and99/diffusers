import os

os.system("""accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  \
  --instance_data_dir="/home/andranik/Documents/Projects/diffusions/diffusers/examples/dreambooth/train_images/train_andranik1"\
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --output_dir="output" \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of sks man" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --seed="0" 
""")
