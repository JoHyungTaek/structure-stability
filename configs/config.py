CFG = {
    "IMG_SIZE": 320,
    "EPOCHS": 15,
    "LEARNING_RATE": 1e-4,
    "BATCH_SIZE": 8,   # mp4까지 쓰면 메모리 많이 먹어서 일단 8 추천
    "SEED": 42,
    "NUM_WORKERS": 2,
    "MODEL_NAME": "efficientnet_b3",
    "DROPOUT": 0.3,
    "WEIGHT_DECAY": 1e-4,
    "N_SPLITS": 5,
    "AMP": True,
    "PATIENCE": 4,
    "TTA": True,

    # video
    "NUM_VIDEO_FRAMES": 6,
    "USE_VIDEO": True,
}

BASE_PATH = "/content/drive/MyDrive/데이콘/open"
MODEL_DIR = "checkpoints"
OOF_PATH = "oof_predictions.csv"
SUBMISSION_PATH = "submission.csv"