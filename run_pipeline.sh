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
python3 src/1_pretrain_text.py --model real --peft qlora --batch_size 1 --steps 60000 --eval_batch 1 --lr 2e-4 --grad_accum_step 4 --warmup_step 3 --save_version 0.1
python3 src/2_finetune_qa.py --peft qlora --batch_size 1 --steps 5 --eval_batch 40000 --lr 2e-4 --grad_accum_step 8 --warmup_step 3 --save_version 0.1 