# for SEED in 1 2 3
# do
#     for RANK in 1 2 4 8 16 32
#     do
#         echo "accelerate launch \
#         --mixed_precision bf16 \
#         --use_deepspeed \
#         --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#         experiments/sequence_classification.py \
#         --task_name sciq \
#         --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#         --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_8/seed_1/ \
#         --seed ${SEED} \
#         --do_train --do_eval --do_pred \
#         --max_seq_length 1024 \
#         --use_lora True --lora_rank ${RANK} \
#         --save_total_limit 1 \
#         --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#         --evaluation_strategy steps --eval_steps 1250 \
#         --save_strategy steps --save_steps 1250 --max_steps 18750 \
#         --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#         --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&"
#     done
# done

RANK=8
for SEED in 1 2 3
do
    for LR in 1e-7
    do
        directory=/net/nfs.cirrascale/allennlp/jacobm/lora-investigation/deberta-v2-xxlarge/case-hold/lora_${RANK}/seed_${SEED}/${LR}/
        mkdir -p ${directory} &&
        accelerate launch \
            --use_deepspeed \
            --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
            experiments/sequence_classification.py \
            --task_name sciq \
            --model_name_or_path microsoft/deberta-v2-xxlarge \
            --output_dir ${directory} \
            --seed ${SEED} \
            --do_train --do_eval --do_predict \
            --max_seq_length 512 \
            --use_lora True --lora_rank ${RANK} \
            --save_total_limit 1 \
            --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
            --evaluation_strategy steps --eval_steps 1250 \
            --save_strategy steps --save_steps 1250 --max_steps 18750 \
            --learning_rate ${LR} --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
            --gradient_accumulation_steps 1 --dataloader_pin_memory False &&
    done
done