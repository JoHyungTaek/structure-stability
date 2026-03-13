CFG = {
    "IMG_SIZE": 320,
    "EPOCHS": 15,
    "LEARNING_RATE": 1e-4,
    "BATCH_SIZE": 16,
    "SEED": 42,
    "NUM_WORKERS": 2,
    "MODEL_NAME": "efficientnet_b3",
    "DROPOUT": 0.3,
    "WEIGHT_DECAY": 1e-4,
    "AMP": True,
    "PATIENCE": 4,
    "TTA": True,
}

BASE_PATH = "/content/drive/MyDrive/데이콘/open"
MODEL_PATH = "best_model.pth"
SUBMISSION_PATH = "submission.csv"