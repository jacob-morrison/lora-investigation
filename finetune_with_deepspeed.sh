MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    experiments/sequence_classification.py \
    --task_name sciq --model_name_or_path gpt2-large \
    --output_dir ../results/ --do_train --do_eval --do_pred --max_seq_length 512 \
    --use_lora True --lora_rank 2 --save_total_limit 1 --load_best_model_at_end \
    --metric_for_best_model accuracy --greater_is_better True --evaluation_strategy steps \
    --eval_steps 1250 --save_strategy steps --save_steps 1250 \
    --max_steps 18750 --learning_rate 1e-4 --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 --seed 1 --gradient_accumulation_steps 1 \
    --dataloader_pin_memory False

accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    experiments/sequence_classification.py \
    --task_name qnli --model_name_or_path google/t5-large-lm-adapt \
    --output_dir ../results/ --do_train --do_eval --max_seq_length 512 \
    --use_lora True --lora_rank 8 --save_total_limit 1 --load_best_model_at_end \
    --metric_for_best_model accuracy --greater_is_better True --evaluation_strategy steps \
    --eval_steps 1250 --save_strategy steps --save_steps 1250 \
    --max_steps 18750 --learning_rate 1e-4 --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 --seed 1 --gradient_accumulation_steps 1 \
    --dataloader_pin_memory False

# accelerate + deepspeed(?)
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    experiments/sequence_classification.py \
    --task_name sciq --model_name_or_path google/t5-large-lm-adapt \
    --output_dir ../results/ --do_train --do_eval --do_pred --max_seq_length 512 \
    --use_lora True --lora_rank 8 --save_total_limit 1 --load_best_model_at_end \
    --metric_for_best_model accuracy --greater_is_better True --evaluation_strategy steps \
    --eval_steps 10 --save_strategy steps --save_steps 10 \
    --max_steps 50 --learning_rate 1e-4 --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 --seed 1 --gradient_accumulation_steps 1 \
    --max_eval_samples 10 --max_train_samples 10 --max_predict_samples 10 \
    --dataloader_pin_memory False
    --bf16 True \
    --use_flash_attn True \

    --mixed_precision bf16 \
# /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/13B 

# pure deepspeed
# deepspeed experiments/sequence_classification.py \
#     --deepspeed ds_configs/stage3_no_offloading.conf \
#     --task_name case-hold --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#     --output_dir ../results/ --do_train --do_eval --do_predict --max_seq_length 1024 \
#     --use_lora True --lora_rank 8 --save_total_limit 1 --load_best_model_at_end \
#     --metric_for_best_model accuracy --greater_is_better True --evaluation_strategy steps \
#     --eval_steps 10 --save_strategy steps --save_steps 10 \
#     --max_steps 50 --learning_rate 1e-4 --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 8 --seed 1 --gradient_accumulation_steps 1 \
#     --max_eval_samples 10 --max_train_samples 10 --max_predict_samples 10 \
#     --bf16 True \
#     --use_flash_attn True \
#     --dataloader_pin_memory False

# /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B # max length 1024++ # 4 train batch size 8 eval batch size for full ft, 8+ train batch for lora
#     --use_flash_attn True \
# google/t5-xxl-lm-adapt # max length 512 # 2 train batch size 8 eval batch size


# open_instruct/finetune_trainer.py \
    # --deepspeed ds_configs/stage3_no_offloading.conf \
    # --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/${MODEL_SIZE} \
    # --tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/${MODEL_SIZE} \
    # --use_fast_tokenizer False \
    # --train_file /net/nfs.cirrascale/allennlp/hamishi/open-instruct/tulu_data/tulu_v1_mix.jsonl \
    # --max_seq_length 512 \
    # --use_lora \
    # --lora_rank 256 \
    # --lora_alpha 256 \
    # --lora_dropout 0.05 \
    # --do_train \
    # --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    # --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    # --learning_rate 2e-5 \
    # --lr_scheduler_type linear \
    # --warmup_ratio 0.03 \
    # --weight_decay 0. \
    # --evaluation_strategy "no" \
    # --logging_steps 1 \
    # --save_strategy epoch \
    # --save_total_limit 1 \
    # --num_train_epochs 3 \
    # --output_dir output/alpaca_${MODEL_SIZE}/ \
    # --bf16 \
    # --tf32 True \
    # --overwrite_output_dir \
    # --report_to "none" \
