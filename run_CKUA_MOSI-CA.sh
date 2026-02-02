
#['a', 't', 'v', 'at', 'av', 'tv', 'atv']
echo "GS+CA: Full run =====  (atv)  ===================="
python -u /root/autodl-tmp/mnt/ztt/CKUA/CKUA-KD-GS-CA.py \
  --dataset=CMUMOSI \
  --audio-feature=wav2vec-large-c-UTT \
  --text-feature=deberta-large-4-UTT \
  --video-feature=manet_UTT \
  --seed=66 \
  --batch-size=128 \
  --epochs=100 \
  --lr=0.002 \
  --hidden=256 \
  --depth=6 \
  --num_heads=8 \
  --drop_rate=0.2 \
  --attn_drop_rate=0.0 \
  --stage_epoch=50 \
  --beta_warmup_frac=0.3 \
  --contrastive_temp=1.0 \
  --contrastive_weight=0.05 \
  --kd_weight=0.1 \
  --beta_kl=0.01 \
  --gpu=0 \
  --kd_temperature=1.0 \
  --test_condition=v \
