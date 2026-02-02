
# CMUMOSI 模态内缺失策略训练脚本

echo "================================================"
echo "Dataset: CMUMOSI"
echo "================================================"

# 基础配置
DATASET="CMUMOSI"
AUDIO_FEATURE="wav2vec-large-c-UTT"
TEXT_FEATURE="deberta-large-4-UTT"
VIDEO_FEATURE="manet_UTT"
SEED=66
BATCH_SIZE=128
EPOCHS=100
LR=0.002
HIDDEN=256
DEPTH=6
NUM_HEADS=8
DROP_RATE=0.2
ATTN_DROP_RATE=0.0
STAGE_EPOCH=50
BETA_WARMUP_FRAC=0.3
CONTRASTIVE_TEMP=1.0
CONTRASTIVE_WEIGHT=0.05
KD_WEIGHT=0.1
BETA_KL=0.01
GPU=1

# 测试不同的缺失比例
MISSING_RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

echo "Starting training with different missing ratios..."
echo "Base configuration:"
echo "  - Dataset: $DATASET"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Epochs: $EPOCHS"
echo "  - Learning rate: $LR"
echo "  - Hidden size: $HIDDEN"
echo "  - Depth: $DEPTH"
echo "  - Missing ratios to test: ${MISSING_RATIOS[*]}"
echo ""

# 运行不同缺失比例的实验
for ratio in "${MISSING_RATIOS[@]}"; do
    echo "================================================"
    echo "Training with missing ratio: $ratio (${ratio}0%)"
    echo "================================================"
    
    python -u /mnt/mydisk/ztt/CKUA/CKUA-KD-GS-CA-ratio.py \
        --dataset=$DATASET \
        --audio-feature=$AUDIO_FEATURE \
        --text-feature=$TEXT_FEATURE \
        --video-feature=$VIDEO_FEATURE \
        --seed=$SEED \
        --batch-size=$BATCH_SIZE \
        --epochs=$EPOCHS \
        --lr=$LR \
        --hidden=$HIDDEN \
        --depth=$DEPTH \
        --num_heads=$NUM_HEADS \
        --drop_rate=$DROP_RATE \
        --attn_drop_rate=$ATTN_DROP_RATE \
        --stage_epoch=$STAGE_EPOCH \
        --beta_warmup_frac=$BETA_WARMUP_FRAC \
        --contrastive_temp=$CONTRASTIVE_TEMP \
        --contrastive_weight=$CONTRASTIVE_WEIGHT \
        --kd_weight=$KD_WEIGHT \
        --beta_kl=$BETA_KL \
        --gpu=$GPU \
        --test_condition=atv \
        --mask_strategy=intra_modal \
        --missing_ratio=$ratio
    
    echo "Completed training with missing ratio: $ratio"
    echo ""
    
    # 等待一段时间再开始下一个实验（可选）
    sleep 10
done

echo "================================================"
echo "All experiments completed!"
echo "Results saved with missing ratio information in filenames"
echo "================================================"