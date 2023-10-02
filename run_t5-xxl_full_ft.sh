# # test run
# accelerate launch \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name sciq \
#     --model_name_or_path google/t5-xxl-lm-adapt \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/t5-xxl/sciq/full_finetuning/seed_1/ \
#     --seed 1 \
#     --do_train --do_eval --do_pred \
#     --max_seq_length 512 \
#     --use_lora False --lora_rank 0 \
#     --save_total_limit 1 \
#     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#     --evaluation_strategy steps --eval_steps 10 \
#     --save_strategy steps --save_steps 20 --max_steps 60 \
#     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 --dataloader_pin_memory False

# # sciq
# accelerate launch \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name sciq \
#     --model_name_or_path google/t5-xxl-lm-adapt \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/t5-xxl/sciq/full_finetuning/seed_1/ \
#     --seed 1 \
#     --do_train --do_eval --do_pred \
#     --max_seq_length 512 \
#     --use_lora False --lora_rank 0 \
#     --save_total_limit 1 \
#     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#     --evaluation_strategy steps --eval_steps 1250 \
#     --save_strategy steps --save_steps 1250 --max_steps 18750 \
#     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 --dataloader_pin_memory False &&

# accelerate launch \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name sciq \
#     --model_name_or_path google/t5-xxl-lm-adapt \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/t5-xxl/sciq/full_finetuning/seed_2/ \
#     --seed 2 \
#     --do_train --do_eval --do_pred \
#     --max_seq_length 512 \
#     --use_lora False --lora_rank 0 \
#     --save_total_limit 1 \
#     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#     --evaluation_strategy steps --eval_steps 1250 \
#     --save_strategy steps --save_steps 1250 --max_steps 18750 \
#     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 --dataloader_pin_memory False &&

# in progress on allennlp cirrascale
accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    experiments/sequence_classification.py \
    --task_name sciq \
    --model_name_or_path google/t5-xxl-lm-adapt \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/t5-xxl/sciq/full_finetuning/seed_3/ \
    --seed 3 \
    --do_train --do_eval --do_pred \
    --max_seq_length 512 \
    --use_lora False --lora_rank 0 \
    --save_total_limit 1 \
    --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
    --evaluation_strategy steps --eval_steps 1250 \
    --save_strategy steps --save_steps 1250 --max_steps 18750 \
    --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 --dataloader_pin_memory False &&

# case hold
accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    experiments/sequence_classification.py \
    --task_name case-hold \
    --model_name_or_path google/t5-xxl-lm-adapt \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/t5-xxl/case-hold/full_finetuning/seed_1/ \
    --seed 1 \
    --do_train --do_eval --do_pred \
    --max_seq_length 512 \
    --use_lora False --lora_rank 0 \
    --save_total_limit 1 \
    --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
    --evaluation_strategy steps --eval_steps 1250 \
    --save_strategy steps --save_steps 1250 --max_steps 18750 \
    --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 --dataloader_pin_memory False &&

accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    experiments/sequence_classification.py \
    --task_name case-hold \
    --model_name_or_path google/t5-xxl-lm-adapt \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/t5-xxl/case-hold/full_finetuning/seed_2/ \
    --seed 2 \
    --do_train --do_eval --do_pred \
    --max_seq_length 512 \
    --use_lora False --lora_rank 0 \
    --save_total_limit 1 \
    --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
    --evaluation_strategy steps --eval_steps 1250 \
    --save_strategy steps --save_steps 1250 --max_steps 18750 \
    --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 --dataloader_pin_memory False &&

accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    experiments/sequence_classification.py \
    --task_name case-hold \
    --model_name_or_path google/t5-xxl-lm-adapt \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/t5-xxl/case-hold/full_finetuning/seed_3/ \
    --seed 3 \
    --do_train --do_eval --do_pred \
    --max_seq_length 512 \
    --use_lora False --lora_rank 0 \
    --save_total_limit 1 \
    --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
    --evaluation_strategy steps --eval_steps 1250 \
    --save_strategy steps --save_steps 1250 --max_steps 18750 \
    --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 --dataloader_pin_memory False &&

# sciq - LoRA
accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    experiments/sequence_classification.py \
    --task_name sciq \
    --model_name_or_path google/t5-xxl-lm-adapt \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/t5-xxl/sciq/lora_8/seed_1/ \
    --seed 1 \
    --do_train --do_eval --do_pred \
    --max_seq_length 512 \
    --use_lora True --lora_rank 8 \
    --save_total_limit 1 \
    --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
    --evaluation_strategy steps --eval_steps 1250 \
    --save_strategy steps --save_steps 1250 --max_steps 18750 \
    --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 --dataloader_pin_memory False &&

accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    experiments/sequence_classification.py \
    --task_name sciq \
    --model_name_or_path google/t5-xxl-lm-adapt \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/t5-xxl/sciq/lora_8/seed_2/ \
    --seed 2 \
    --do_train --do_eval --do_pred \
    --max_seq_length 512 \
    --use_lora True --lora_rank 8 \
    --save_total_limit 1 \
    --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
    --evaluation_strategy steps --eval_steps 1250 \
    --save_strategy steps --save_steps 1250 --max_steps 18750 \
    --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 --dataloader_pin_memory False &&

accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    experiments/sequence_classification.py \
    --task_name sciq \
    --model_name_or_path google/t5-xxl-lm-adapt \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/t5-xxl/sciq/lora_8/seed_3/ \
    --seed 3 \
    --do_train --do_eval --do_pred \
    --max_seq_length 512 \
    --use_lora True --lora_rank 8 \
    --save_total_limit 1 \
    --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
    --evaluation_strategy steps --eval_steps 1250 \
    --save_strategy steps --save_steps 1250 --max_steps 18750 \
    --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 --dataloader_pin_memory False &&

# case hold
accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    experiments/sequence_classification.py \
    --task_name case-hold \
    --model_name_or_path google/t5-xxl-lm-adapt \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/t5-xxl/case-hold/lora_8/seed_1/ \
    --seed 1 \
    --do_train --do_eval --do_pred \
    --max_seq_length 512 \
    --use_lora True --lora_rank 8 \
    --save_total_limit 1 \
    --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
    --evaluation_strategy steps --eval_steps 1250 \
    --save_strategy steps --save_steps 1250 --max_steps 18750 \
    --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 --dataloader_pin_memory False &&

accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    experiments/sequence_classification.py \
    --task_name case-hold \
    --model_name_or_path google/t5-xxl-lm-adapt \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/t5-xxl/case-hold/lora_8/seed_2/ \
    --seed 2 \
    --do_train --do_eval --do_pred \
    --max_seq_length 512 \
    --use_lora True --lora_rank 8 \
    --save_total_limit 1 \
    --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
    --evaluation_strategy steps --eval_steps 1250 \
    --save_strategy steps --save_steps 1250 --max_steps 18750 \
    --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 --dataloader_pin_memory False &&

accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    experiments/sequence_classification.py \
    --task_name case-hold \
    --model_name_or_path google/t5-xxl-lm-adapt \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/t5-xxl/case-hold/lora_8/seed_3/ \
    --seed 3 \
    --do_train --do_eval --do_pred \
    --max_seq_length 512 \
    --use_lora True --lora_rank 8 \
    --save_total_limit 1 \
    --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
    --evaluation_strategy steps --eval_steps 1250 \
    --save_strategy steps --save_steps 1250 --max_steps 18750 \
    --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 --dataloader_pin_memory False

    

    