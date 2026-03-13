CFG = {
    "BASE_PATH": "/content/drive/MyDrive/데이콘/open",
    "SEED": 42,
    "NUM_WORKERS": 2,
    "AMP": True,

    "MODEL_NAME": "efficientnet_b3",
    "IMG_SIZE": 384,
    "DROPOUT": 0.4,

    # Stage1: train -> dev
    "STAGE1_EPOCHS": 10,
    "STAGE1_BATCH_SIZE": 8,
    "STAGE1_LR": 5e-5,
    "STAGE1_WEIGHT_DECAY": 2e-4,
    "STAGE1_PATIENCE": 3,

    # Stage2 split
    "DEV_VALID_RATIO": 0.25,

    # Stage2-A: head only adaptation
    "STAGE2_HEAD_EPOCHS": 4,
    "STAGE2_HEAD_BATCH_SIZE": 8,
    "STAGE2_HEAD_LR": 3e-4,
    "STAGE2_HEAD_WEIGHT_DECAY": 1e-4,
    "STAGE2_HEAD_PATIENCE": 2,

    # Stage2-B: last blocks only finetune
    "STAGE2_FINE_EPOCHS": 6,
    "STAGE2_FINE_BATCH_SIZE": 8,
    "STAGE2_FINE_LR": 5e-6,
    "STAGE2_FINE_WEIGHT_DECAY": 1e-4,
    "STAGE2_FINE_PATIENCE": 2,
    "UNFREEZE_LAST_N_BLOCKS": 2,

    # Refit all dev
    "REFIT_EPOCHS": 4,
    "REFIT_BATCH_SIZE": 8,
    "REFIT_LR": 5e-6,
    "REFIT_WEIGHT_DECAY": 1e-4,

    # Inference
    "TTA": True,
}

PRETRAIN_MODEL_PATH = "stage1_pretrain_best.pth"
STAGE2_HEAD_MODEL_PATH = "stage2_head_best.pth"
FINETUNE_MODEL_PATH = "stage2_finetune_best.pth"
FINAL_MODEL_PATH = "final_model_refit_dev_all.pth"
SUBMISSION_PATH = "submission.csv"