# # lora 8
# # accelerate launch \
# #     --mixed_precision bf16 \
# #     --use_deepspeed \
# #     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
# #     experiments/sequence_classification.py \
# #     --task_name sciq \
# #     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
# #     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_1/seed_1/ \
# #     --seed 1 \
# #     --do_train --do_eval --do_pred \
# #     --max_seq_length 1024 \
# #     --use_lora True --lora_rank 1 \
# #     --save_total_limit 1 \
# #     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
# #     --evaluation_strategy steps --eval_steps 1250 \
# #     --save_strategy steps --save_steps 1250 --max_steps 18750 \
# #     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
# #     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# # accelerate launch \
# #     --mixed_precision bf16 \
# #     --use_deepspeed \
# #     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
# #     experiments/sequence_classification.py \
# #     --task_name sciq \
# #     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
# #     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_1/seed_2/ \
# #     --seed 2 \
# #     --do_train --do_eval --do_pred \
# #     --max_seq_length 1024 \
# #     --use_lora True --lora_rank 1 \
# #     --save_total_limit 1 \
# #     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
# #     --evaluation_strategy steps --eval_steps 1250 \
# #     --save_strategy steps --save_steps 1250 --max_steps 18750 \
# #     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
# #     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# # in progress on ib 46
# # accelerate launch \
# #     --mixed_precision bf16 \
# #     --use_deepspeed \
# #     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
# #     experiments/sequence_classification.py \
# #     --task_name sciq \
# #     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
# #     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_1/seed_3/ \
# #     --seed 3 \
# #     --do_train --do_eval --do_pred \
# #     --max_seq_length 1024 \
# #     --use_lora True --lora_rank 1 \
# #     --save_total_limit 1 \
# #     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
# #     --evaluation_strategy steps --eval_steps 1250 \
# #     --save_strategy steps --save_steps 1250 --max_steps 18750 \
# #     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
# #     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# # # lora 2
# # accelerate launch \
# #     --mixed_precision bf16 \
# #     --use_deepspeed \
# #     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
# #     experiments/sequence_classification.py \
# #     --task_name sciq \
# #     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
# #     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_2/seed_1/ \
# #     --seed 1 \
# #     --do_train --do_eval --do_pred \
# #     --max_seq_length 1024 \
# #     --use_lora True --lora_rank 2 \
# #     --save_total_limit 1 \
# #     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
# #     --evaluation_strategy steps --eval_steps 1250 \
# #     --save_strategy steps --save_steps 1250 --max_steps 18750 \
# #     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
# #     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# # accelerate launch \
# #     --mixed_precision bf16 \
# #     --use_deepspeed \
# #     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
# #     experiments/sequence_classification.py \
# #     --task_name sciq \
# #     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
# #     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_2/seed_2/ \
# #     --seed 2 \
# #     --do_train --do_eval --do_pred \
# #     --max_seq_length 1024 \
# #     --use_lora True --lora_rank 2 \
# #     --save_total_limit 1 \
# #     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
# #     --evaluation_strategy steps --eval_steps 1250 \
# #     --save_strategy steps --save_steps 1250 --max_steps 18750 \
# #     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
# #     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# # accelerate launch \
# #     --mixed_precision bf16 \
# #     --use_deepspeed \
# #     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
# #     experiments/sequence_classification.py \
# #     --task_name sciq \
# #     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
# #     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_2/seed_3/ \
# #     --seed 3 \
# #     --do_train --do_eval --do_pred \
# #     --max_seq_length 1024 \
# #     --use_lora True --lora_rank 2 \
# #     --save_total_limit 1 \
# #     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
# #     --evaluation_strategy steps --eval_steps 1250 \
# #     --save_strategy steps --save_steps 1250 --max_steps 18750 \
# #     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
# #     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# # # lora 4
# # accelerate launch \
# #     --mixed_precision bf16 \
# #     --use_deepspeed \
# #     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
# #     experiments/sequence_classification.py \
# #     --task_name sciq \
# #     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
# #     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_4/seed_1/ \
# #     --seed 1 \
# #     --do_train --do_eval --do_pred \
# #     --max_seq_length 1024 \
# #     --use_lora True --lora_rank 4 \
# #     --save_total_limit 1 \
# #     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
# #     --evaluation_strategy steps --eval_steps 1250 \
# #     --save_strategy steps --save_steps 1250 --max_steps 18750 \
# #     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
# #     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# # accelerate launch \
# #     --mixed_precision bf16 \
# #     --use_deepspeed \
# #     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
# #     experiments/sequence_classification.py \
# #     --task_name sciq \
# #     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
# #     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_4/seed_2/ \
# #     --seed 2 \
# #     --do_train --do_eval --do_pred \
# #     --max_seq_length 1024 \
# #     --use_lora True --lora_rank 4 \
# #     --save_total_limit 1 \
# #     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
# #     --evaluation_strategy steps --eval_steps 1250 \
# #     --save_strategy steps --save_steps 1250 --max_steps 18750 \
# #     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
# #     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# # accelerate launch \
# #     --mixed_precision bf16 \
# #     --use_deepspeed \
# #     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
# #     experiments/sequence_classification.py \
# #     --task_name sciq \
# #     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
# #     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_4/seed_3/ \
# #     --seed 3 \
# #     --do_train --do_eval --do_pred \
# #     --max_seq_length 1024 \
# #     --use_lora True --lora_rank 4 \
# #     --save_total_limit 1 \
# #     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
# #     --evaluation_strategy steps --eval_steps 1250 \
# #     --save_strategy steps --save_steps 1250 --max_steps 18750 \
# #     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
# #     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# # # lora 16

# # accelerate launch \
# #     --mixed_precision bf16 \
# #     --use_deepspeed \
# #     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
# #     experiments/sequence_classification.py \
# #     --task_name sciq \
# #     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
# #     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_16/seed_1/ \
# #     --seed 1 \
# #     --do_train --do_eval --do_pred \
# #     --max_seq_length 1024 \
# #     --use_lora True --lora_rank 16 \
# #     --save_total_limit 1 \
# #     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
# #     --evaluation_strategy steps --eval_steps 1250 \
# #     --save_strategy steps --save_steps 1250 --max_steps 18750 \
# #     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
# #     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# # accelerate launch \
# #     --mixed_precision bf16 \
# #     --use_deepspeed \
# #     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
# #     experiments/sequence_classification.py \
# #     --task_name sciq \
# #     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
# #     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_16/seed_2/ \
# #     --seed 2 \
# #     --do_train --do_eval --do_pred \
# #     --max_seq_length 1024 \
# #     --use_lora True --lora_rank 16 \
# #     --save_total_limit 1 \
# #     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
# #     --evaluation_strategy steps --eval_steps 1250 \
# #     --save_strategy steps --save_steps 1250 --max_steps 18750 \
# #     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
# #     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# # accelerate launch \
# #     --mixed_precision bf16 \
# #     --use_deepspeed \
# #     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
# #     experiments/sequence_classification.py \
# #     --task_name sciq \
# #     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
# #     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_16/seed_3/ \
# #     --seed 3 \
# #     --do_train --do_eval --do_pred \
# #     --max_seq_length 1024 \
# #     --use_lora True --lora_rank 16 \
# #     --save_total_limit 1 \
# #     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
# #     --evaluation_strategy steps --eval_steps 1250 \
# #     --save_strategy steps --save_steps 1250 --max_steps 18750 \
# #     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
# #     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# # # lora 32

# # accelerate launch \
# #     --mixed_precision bf16 \
# #     --use_deepspeed \
# #     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
# #     experiments/sequence_classification.py \
# #     --task_name sciq \
# #     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
# #     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_32/seed_1/ \
# #     --seed 1 \
# #     --do_train --do_eval --do_pred \
# #     --max_seq_length 1024 \
# #     --use_lora True --lora_rank 32 \
# #     --save_total_limit 1 \
# #     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
# #     --evaluation_strategy steps --eval_steps 1250 \
# #     --save_strategy steps --save_steps 1250 --max_steps 18750 \
# #     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
# #     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# # accelerate launch \
# #     --mixed_precision bf16 \
# #     --use_deepspeed \
# #     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
# #     experiments/sequence_classification.py \
# #     --task_name sciq \
# #     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
# #     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_32/seed_2/ \
# #     --seed 2 \
# #     --do_train --do_eval --do_pred \
# #     --max_seq_length 1024 \
# #     --use_lora True --lora_rank 32 \
# #     --save_total_limit 1 \
# #     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
# #     --evaluation_strategy steps --eval_steps 1250 \
# #     --save_strategy steps --save_steps 1250 --max_steps 18750 \
# #     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
# #     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# # accelerate launch \
# #     --mixed_precision bf16 \
# #     --use_deepspeed \
# #     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
# #     experiments/sequence_classification.py \
# #     --task_name sciq \
# #     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
# #     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_32/seed_3/ \
# #     --seed 3 \
# #     --do_train --do_eval --do_pred \
# #     --max_seq_length 1024 \
# #     --use_lora True --lora_rank 32 \
# #     --save_total_limit 1 \
# #     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
# #     --evaluation_strategy steps --eval_steps 1250 \
# #     --save_strategy steps --save_steps 1250 --max_steps 18750 \
# #     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
# #     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# # lora 2521

# accelerate launch \
#     --mixed_precision bf16 \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name sciq \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_2521/seed_1/ \
#     --seed 1 \
#     --do_train --do_eval --do_pred \
#     --max_seq_length 1024 \
#     --use_lora True --lora_rank 2521 \
#     --save_total_limit 1 \
#     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#     --evaluation_strategy steps --eval_steps 1250 \
#     --save_strategy steps --save_steps 1250 --max_steps 18750 \
#     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# accelerate launch \
#     --mixed_precision bf16 \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name sciq \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_2521/seed_2/ \
#     --seed 2 \
#     --do_train --do_eval --do_pred \
#     --max_seq_length 1024 \
#     --use_lora True --lora_rank 2521 \
#     --save_total_limit 1 \
#     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#     --evaluation_strategy steps --eval_steps 1250 \
#     --save_strategy steps --save_steps 1250 --max_steps 18750 \
#     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# accelerate launch \
#     --mixed_precision bf16 \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name sciq \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_2521/seed_3/ \
#     --seed 3 \
#     --do_train --do_eval --do_pred \
#     --max_seq_length 1024 \
#     --use_lora True --lora_rank 2521 \
#     --save_total_limit 1 \
#     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#     --evaluation_strategy steps --eval_steps 1250 \
#     --save_strategy steps --save_steps 1250 --max_steps 18750 \
#     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True && 

# # lora 5042

# accelerate launch \
#     --mixed_precision bf16 \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name sciq \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_5042/seed_1/ \
#     --seed 1 \
#     --do_train --do_eval --do_pred \
#     --max_seq_length 1024 \
#     --use_lora True --lora_rank 5042 \
#     --save_total_limit 1 \
#     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#     --evaluation_strategy steps --eval_steps 1250 \
#     --save_strategy steps --save_steps 1250 --max_steps 18750 \
#     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# accelerate launch \
#     --mixed_precision bf16 \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name sciq \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_5042/seed_2/ \
#     --seed 2 \
#     --do_train --do_eval --do_pred \
#     --max_seq_length 1024 \
#     --use_lora True --lora_rank 5042 \
#     --save_total_limit 1 \
#     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#     --evaluation_strategy steps --eval_steps 1250 \
#     --save_strategy steps --save_steps 1250 --max_steps 18750 \
#     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# accelerate launch \
#     --mixed_precision bf16 \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name sciq \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_5042/seed_3/ \
#     --seed 3 \
#     --do_train --do_eval --do_pred \
#     --max_seq_length 1024 \
#     --use_lora True --lora_rank 5042 \
#     --save_total_limit 1 \
#     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#     --evaluation_strategy steps --eval_steps 1250 \
#     --save_strategy steps --save_steps 1250 --max_steps 18750 \
#     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True && 

# # lora 7562

# accelerate launch \
#     --mixed_precision bf16 \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name sciq \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_7562/seed_1/ \
#     --seed 1 \
#     --do_train --do_eval --do_pred \
#     --max_seq_length 1024 \
#     --use_lora True --lora_rank 7562 \
#     --save_total_limit 1 \
#     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#     --evaluation_strategy steps --eval_steps 1250 \
#     --save_strategy steps --save_steps 1250 --max_steps 18750 \
#     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# accelerate launch \
#     --mixed_precision bf16 \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name sciq \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_7562/seed_2/ \
#     --seed 2 \
#     --do_train --do_eval --do_pred \
#     --max_seq_length 1024 \
#     --use_lora True --lora_rank 7562 \
#     --save_total_limit 1 \
#     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#     --evaluation_strategy steps --eval_steps 1250 \
#     --save_strategy steps --save_steps 1250 --max_steps 18750 \
#     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# accelerate launch \
#     --mixed_precision bf16 \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name sciq \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_7562/seed_3/ \
#     --seed 3 \
#     --do_train --do_eval --do_pred \
#     --max_seq_length 1024 \
#     --use_lora True --lora_rank 7562 \
#     --save_total_limit 1 \
#     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#     --evaluation_strategy steps --eval_steps 1250 \
#     --save_strategy steps --save_steps 1250 --max_steps 18750 \
#     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True && 

# # lora 10083

# accelerate launch \
#     --mixed_precision bf16 \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name sciq \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_10083/seed_1/ \
#     --seed 1 \
#     --do_train --do_eval --do_pred \
#     --max_seq_length 1024 \
#     --use_lora True --lora_rank 10083 \
#     --save_total_limit 1 \
#     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#     --evaluation_strategy steps --eval_steps 1250 \
#     --save_strategy steps --save_steps 1250 --max_steps 18750 \
#     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# accelerate launch \
#     --mixed_precision bf16 \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name sciq \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_10083/seed_2/ \
#     --seed 2 \
#     --do_train --do_eval --do_pred \
#     --max_seq_length 1024 \
#     --use_lora True --lora_rank 10083 \
#     --save_total_limit 1 \
#     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#     --evaluation_strategy steps --eval_steps 1250 \
#     --save_strategy steps --save_steps 1250 --max_steps 18750 \
#     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

# accelerate launch \
#     --mixed_precision bf16 \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name sciq \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_10083/seed_3/ \
#     --seed 3 \
#     --do_train --do_eval --do_pred \
#     --max_seq_length 1024 \
#     --use_lora True --lora_rank 10083 \
#     --save_total_limit 1 \
#     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#     --evaluation_strategy steps --eval_steps 1250 \
#     --save_strategy steps --save_steps 1250 --max_steps 18750 \
#     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True && 

# lora 12603

# accelerate launch \
#     --mixed_precision bf16 \
#     --use_deepspeed \
#     --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
#     experiments/sequence_classification.py \
#     --task_name sciq \
#     --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
#     --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_12603/seed_1/ \
#     --seed 1 \
#     --do_train --do_eval --do_pred \
#     --max_seq_length 1024 \
#     --use_lora True --lora_rank 12603 \
#     --save_total_limit 1 \
#     --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
#     --evaluation_strategy steps --eval_steps 1250 \
#     --save_strategy steps --save_steps 1250 --max_steps 18750 \
#     --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
#     --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

accelerate launch \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    experiments/sequence_classification.py \
    --task_name sciq \
    --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_12603/seed_2/ \
    --seed 2 \
    --do_train --do_eval --do_pred \
    --max_seq_length 1024 \
    --use_lora True --lora_rank 12603 \
    --save_total_limit 1 \
    --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
    --evaluation_strategy steps --eval_steps 313 \
    --save_strategy steps --save_steps 313 --max_steps 4688 \
    --learning_rate 1e-4 --per_device_train_batch_size 4 --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True &&

accelerate launch \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    experiments/sequence_classification.py \
    --task_name sciq \
    --model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B \
    --output_dir /net/nfs.cirrascale/allennlp/jacobm/lora-investigation/llama2-7b/sciq/lora_12603/seed_3/ \
    --seed 3 \
    --do_train --do_eval --do_pred \
    --max_seq_length 1024 \
    --use_lora True --lora_rank 12603 \
    --save_total_limit 1 \
    --load_best_model_at_end --metric_for_best_model accuracy --greater_is_better True \
    --evaluation_strategy steps --eval_steps 313 \
    --save_strategy steps --save_steps 313 --max_steps 4688 \
    --learning_rate 1e-4 --per_device_train_batch_size 1 --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 --dataloader_pin_memory False --bf16 True --use_flash_attn True
