CFG = {
    "IMG_SIZE": 380,
    "EPOCHS": 12,
    "LEARNING_RATE": 5e-5,
    "BATCH_SIZE": 8,
    "ACCUM_STEPS": 2,
    "SEED": 42,
    "NUM_WORKERS": 2,
    "MODEL_NAME": "efficientnet_b4",
    "DROPOUT": 0.4,
    "WEIGHT_DECAY": 2e-4,
    "AMP": True,
    "PATIENCE": 3,
    "TTA": True,
    "LABEL_SMOOTHING": 0.02,
}

BASE_PATH = "/content/drive/MyDrive/데이콘/open"
MODEL_PATH = "best_model.pth"
SUBMISSION_PATH = "submission.csv"