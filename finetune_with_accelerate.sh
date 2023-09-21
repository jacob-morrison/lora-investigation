export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=8
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# accelerate launch \
#     --mixed_precision bf16 \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name case-hold \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/${MODEL_SIZE} \
#     --use_flash_attn \
#     --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/${MODEL_SIZE} \
#     --output_dir ../output/ \
#     --do_train \
#     --do_eval \
#     --do_pred \
#     --bf16 \
#     --overwrite_output_dir \
#     --max_seq_length 1024 \
#     --use_lora False \
#     --lora_rank 8 \
#     --save_total_limit 5 \
#     --load_best_model_at_end \
#     --metric_for_best_model accuracy \
#     --greater_is_better True \
#     --evaluation_strategy steps \
#     --eval_steps 125 \
#     --save_strategy steps \
#     --save_steps 1250 \
#     --max_steps 18750 \
#     --learning_rate 5e-5 \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --per_device_eval_batch_size 8 \
#     --seed 1 \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --dataloader_pin_memory False

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 8 \
    experiments/sequence_classification.py \
    --task_name case-hold --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
    --output_dir ../results/ --do_train --do_eval --do_predict --max_seq_length 1024 \
    --use_lora True --lora_rank 8 --save_total_limit 1 --load_best_model_at_end \
    --metric_for_best_model accuracy --greater_is_better True --evaluation_strategy steps \
    --eval_steps 10 --save_strategy steps --save_steps 10 \
    --max_steps 50 --learning_rate 1e-4 --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 --seed 1 --gradient_accumulation_steps 1 \
    --max_eval_samples 10 --max_train_samples 10 --max_predict_samples 10 \
    --use_flash_attn True \
    --gradient_checkpointing False \
    --dataloader_pin_memory False

# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes 8 \
#     experiments/sequence_classification.py \
#     --task_name case-hold --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#     --output_dir ../results/ --do_train --do_eval --do_predict --max_seq_length 1024 \
#     --use_lora True --lora_rank 8 --save_total_limit 1 --load_best_model_at_end \
#     --metric_for_best_model accuracy --greater_is_better True --evaluation_strategy steps \
#     --eval_steps 1250 --save_strategy steps --save_steps 1250 \
#     --max_steps 18750 --learning_rate 1e-4 --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 8 --seed 1 --gradient_accumulation_steps 1 \
#     --dataloader_pin_memory False


# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes 8 \
#     experiments/sequence_classification_no_trainer.py \
#     --task_name case-hold --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#     --output_dir ../results/ --do_train --do_eval --do_predict --max_seq_length 1024 \
#     --use_lora True --lora_rank 8 --save_total_limit 1 --load_best_model_at_end \
#     --metric_for_best_model accuracy --greater_is_better True --evaluation_strategy steps \
#     --eval_steps 10 --save_strategy steps --checkpointing_steps 1000 \
#     --max_train_steps 50 --learning_rate 1e-4 --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 8 --seed 1 --gradient_accumulation_steps 1 \
#     --max_eval_samples 10 --max_train_samples 10 --max_predict_samples 10 \
#     --dataloader_pin_memory False