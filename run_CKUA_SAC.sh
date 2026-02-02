
# 支持两种缺失策略：模态内缺失(intra_modal) 和 模态间缺失(inter_modal)

echo "SW: 模态间缺失策略测试 ===========  (atv)  ========="
python -u /root/autodl-tmp/mnt/ztt/CKUA/CKUA-KD-GS-CA_SAC.py \
  --dataset=SAC \
  --seed=66 \
  --batch-size=128 \
  --epochs=100 \
  --lr=0.0002 \
  --hidden=256 \
  --depth=6 \
  --num_heads=8 \
  --drop_rate=0.2 \
  --attn_drop_rate=0.0 \
  --test_condition=atv \
  --stage_epoch=30 \
  --beta_warmup_frac=0.3 \
  --contrastive_temp=0.1 \
  --contrastive_weight=0.1 \
  --kd_weight=0.1 \
  --beta_kl=0.1 \
  --l2=0.0001 \
  --gpu=0 \
  --kd_temperature=1.0 \
  --mask_strategy=inter_modal