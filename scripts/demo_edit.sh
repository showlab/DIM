export TOKENIZERS_PARALLELISM=false

python3 ./infer/demo_edit.py \
    --pretrained_model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --model_name_or_path ./checkpoints/DIM-4.6B-Edit \
    --condition_type 'LMToken' \
    --max_condition_length 8192 \
    --sana_config ./models/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5_channel_cond.yaml \
    --with_latents_condition True \
    --text_only_condition False \
    --dataset_type TosDatasetEdit \
    --dataset_path ./cache/demo/tos_dataset_edit_demo.jsonl \
    --sample_size 1 \
    --gen_resolution 1024 \
    --force_gen_resolution True \
    --task_type MM-PAD \
    --num_chunk 1 \
    --chunk_idx 0
