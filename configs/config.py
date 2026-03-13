CFG = {
    # data
    "BASE_PATH": "/content/drive/MyDrive/데이콘/open",

    # runtime
    "SEED": 42,
    "NUM_WORKERS": 2,
    "AMP": True,

    # model
    "MODEL_NAME": "efficientnet_b3",
    "IMG_SIZE": 384,
    "DROPOUT": 0.4,

    # Stage1 : train → dev
    "STAGE1_EPOCHS": 10,
    "STAGE1_BATCH_SIZE": 8,
    "STAGE1_LR": 5e-5,
    "STAGE1_WEIGHT_DECAY": 2e-4,
    "STAGE1_PATIENCE": 3,

    # Stage2 : dev split finetune
    "STAGE2_EPOCHS": 8,
    "STAGE2_BATCH_SIZE": 8,
    "STAGE2_LR": 2e-5,
    "STAGE2_WEIGHT_DECAY": 1e-4,
    "STAGE2_PATIENCE": 3,
    "DEV_VALID_RATIO": 0.2,

    # refit
    "REFIT_EPOCHS": 4,
    "REFIT_BATCH_SIZE": 8,
    "REFIT_LR": 1e-5,
    "REFIT_WEIGHT_DECAY": 1e-4,

    # inference
    "TTA": True,
}

PRETRAIN_MODEL_PATH = "stage1_pretrain_best.pth"
FINETUNE_MODEL_PATH = "stage2_finetune_best.pth"
FINAL_MODEL_PATH = "final_model_refit_dev_all.pth"
SUBMISSION_PATH = "submission.csv"