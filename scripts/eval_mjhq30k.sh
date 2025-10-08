export TOKENIZERS_PARALLELISM=false

GPU_NUM=8

for (( GPU_ID=0; GPU_ID<GPU_NUM; GPU_ID++ )); do
  CUDA_VISIBLE_DEVICES=$GPU_ID python3 ./infer/eval_mjhq30k.py \
    --pretrained_model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --model_name_or_path ./checkpoints/DIM-4.6B-T2I \
    --condition_type 'LMToken' \
    --max_condition_length 8192 \
    --sana_config ./models/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml \
    --with_latents_condition False \
    --text_only_condition False \
    --dataset_type TosDatasetBase \
    --dataset_path ./cache/MJHQ-30K/tos_dataset.jsonl \
    --sample_size 1 \
    --gen_resolution 1024 \
    --force_gen_resolution True \
    --task_type T2I-NPAD \
    --num_chunk $GPU_NUM \
    --chunk_idx $GPU_ID &
done
wait