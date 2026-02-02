#!/bin/bash
echo ""
echo "Testing intra-modal missing strategy..."

# 定义要测试的缺失率
missing_ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# 循环运行每种缺失率
for ratio in "${missing_ratios[@]}"; do
    echo "Run MEPSA ======== (missing_ratio=$ratio) ===================="
    
    # 模态内缺失策略（新的逻辑）
    python -u /mnt/mydisk/ztt/CKUA/CKUA-KD-GS-CA_MEPSA.py \
      --dataset=MEPSA \
      --seed=66 \
      --batch-size=128 \
      --epochs=100 \
      --lr=0.0002 \
      --hidden=256 \
      --depth=6 \
      --num_heads=8 \
      --drop_rate=0.2 \
      --attn_drop_rate=0.0 \
      --stage_epoch=30 \
      --beta_warmup_frac=0.3 \
      --contrastive_temp=0.1 \
      --contrastive_weight=0.1 \
      --kd_weight=0.1 \
      --beta_kl=0.1 \
      --gpu=0 \
      --test_condition=atv \
      --mask_strategy=intra_modal \
      --missing_ratio=$ratio \
    
    echo "Completed missing_ratio: $ratio"
    echo "=========================================="
done

echo "All missing ratios completed!"