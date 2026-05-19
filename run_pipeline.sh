#===============================================================
#conda activate env
#===============================================================
# save_version should be increaase exact combination training 
#===============================================================================================
#FOR SMALLER VRAM GPU SMALLER MODEL FFT 
#===============================================================================================
# python3 src/1_pretrain_text.py --model real --peft fft --batch_size 1 --steps 100000 --eval_batch 1 --lr 2e-4 --grad_accum_step 4 --warmup_step 3 --save_version 0.1

#FOR SMALLER VRAM GPU SMALLER MODEL LoRA 
#===============================================================================================
#python3 train_step1.py --model debug --peft lora --batch-size 8 --epochs 20 --eval_batch 2 --lr 2e-4 --grad_accum_step 8 --warmup_step 3 --save_version 0.1

#FOR SMALLER VRAM GPU SMALLER MODEL QLoRA 
#===============================================================================================
#python3 train_step1.py --model debug --peft lora --batch-size 8 --epochs 20 --eval_batch 2 --lr 2e-4 --grad_accum_step 8 --warmup_step 3 --save_version 0.1

#===============================================================================================
#FOR WORKSTATION GPU BIGGER MODEL FFT 
#===============================================================================================
#python3 train_step1.py --model real --peft fft --batch-size 1 --epochs 20 --eval_batch 1 --lr 2e-4 --grad_accum_step 4 --warmup_step 3 --save_version 0.1

#===============================================================================================
#FOR WORKSTATION GPU BIGGER MODEL LoRA 
#===============================================================================================
#python3 train_step1.py --model real --peft lora --batch-size 8 --epochs 20 --eval_batch 2 --lr 2e-4 --grad_accum_step 8 --warmup_step 3 --save_version 0.1

#===============================================================================================
#FOR WORKSTATION GPU BIGGER MODEL QLoRA 
#===============================================================================================
#python3 train_step1.py --model real --peft qlora --batch-size 8 --epochs 20 --eval_batch 2 --lr 2e-4 --grad_accum_step 8 --warmup_step 3 --save_version 0.1



# For LLM LAB training 
# ~4 epochs Trained ~51 hours 
#python3 src/1_pretrain_text.py --model real --peft qlora --batch_size 2 --steps 25000 --eval_batch 1 --lr 2e-4 --grad_accum_step 4 --warmup_step 700 --save_version 0.1 
# ~6 epochs Trained ~1.3 hour
python3 src/2_finetune_qa.py --peft qlora --batch_size 4 --epochs 3 --eval_batch 2 --lr 3e-5 --grad_accum_step 4 --warmup_step 2000 --save_version 0.3

# python3 src/2_finetune_qa.py --peft qlora --batch_size 6 --steps 5 --eval_batch 2 --lr 3e-5 --grad_accum_step 4 --warmup_step 150 --save_version 0.3


