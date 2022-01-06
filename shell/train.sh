export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch --gpus "0" train.py \
    --model_type convbert \
    --model_name_or_path convbert-base \
    --task_name sst-2 \
    --max_seq_length 128 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --output_dir ./convbert_base_outputs/ \
    --logging_steps 100 \
    --save_steps 400 \
    --batch_size 32   \
    --warmup_proportion 0.1
