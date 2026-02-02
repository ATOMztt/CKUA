# CMUMOSEI

test_conditions=("a" "t" "v" "at" "av" "tv" "atv")

# 循环运行每种测试条件
for condition in "${test_conditions[@]}"; do
    echo ""
    echo "GS+CA: Full run ======== (test_condition=$condition) ===================="
    python -u /root/autodl-tmp/mnt/ztt/CKUA/CKUA-KD-GS-CA.py \
      --dataset=CMUMOSEI \
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
      --drop_rate=0.5 \
      --attn_drop_rate=0.0 \
      --stage_epoch=10 \
      --beta_warmup_frac=0.3 \
      --contrastive_temp=0.5 \
      --contrastive_weight=0.05 \
      --kd_weight=1.0 \
      --beta_kl=0.005 \
      --gpu=0 \
      --test_condition=$condition

done
