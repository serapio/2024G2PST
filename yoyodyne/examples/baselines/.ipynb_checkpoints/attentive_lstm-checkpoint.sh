EXPERIMENT="G2P-LSTM-${LANGUAGE}"
TRAIN="2024G2PST/data/tsv/${SCRIPT}/${LANGUAGE}/train/${LANGUAGE}_train.tsv"
DEV="2024G2PST/data/tsv/${SCRIPT}/${LANGUAGE}/val/${LANGUAGE}_val.tsv"

yoyodyne-train \
	--model_dir="~/models" \
	--experiment "${EXPERIMENT}" \
    --train="${TRAIN}" \
    --val="${DEV}" \
    --arch "attentive_lstm" \
	--source_encoder_arch="lstm" \
    --features_encoder_arch "linear" \
    --embedding_size 256 \
    --hidden_size 1024 \
    --encoder_layers 4 \
    --decoder_layers 1 \
	--source_attention_heads 4 \
    --batch_size 32 \
    --dropout .1 \
    --check_val_every_n_epoch 1 \
    --log_every_n_step 10 \
    --gradient_clip_val 3 \
    --label_smoothing .1 \
    --optimizer adam \
    --learning_rate .001 \
	--max_steps 1000 \
    --seed 49 \
    --accelerator gpu \
    --precision 16 \
	--features_col 0 \
    --check_val_every_n_epoch 1 \
    --log_wandb