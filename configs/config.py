CFG = {
    # common
    "BASE_PATH": "/content/drive/MyDrive/데이콘/open",
    "SEED": 42,
    "NUM_WORKERS": 2,
    "AMP": True,

    # teacher
    "TEACHER_MODEL_NAME": "efficientnet_b3",
    "TEACHER_IMG_SIZE": 320,
    "TEACHER_EPOCHS": 12,
    "TEACHER_BATCH_SIZE": 8,
    "TEACHER_LR": 1e-4,
    "TEACHER_WEIGHT_DECAY": 1e-4,
    "TEACHER_DROPOUT": 0.3,
    "NUM_VIDEO_FRAMES": 6,
    "TEACHER_PATIENCE": 3,

    # student
    "STUDENT_MODEL_NAME": "convnext_small",
    "STUDENT_IMG_SIZE": 384,
    "STUDENT_EPOCHS": 14,
    "STUDENT_BATCH_SIZE": 12,
    "STUDENT_LR": 8e-5,
    "STUDENT_WEIGHT_DECAY": 1e-4,
    "STUDENT_DROPOUT": 0.3,
    "STUDENT_PATIENCE": 4,

    # distillation
    "DISTILL_ALPHA": 0.35,   # soft loss 가중치
    "DISTILL_TEMP": 2.0,

    # ensemble / inference
    "N_SEEDS": 3,
    "SEEDS": [42, 52, 62],
    "TTA": True,
}

TEACHER_MODEL_PATH = "teacher_best.pth"
SOFT_LABEL_PATH = "train_soft_labels.csv"
STUDENT_MODEL_DIR = "student_checkpoints"
SUBMISSION_PATH = "submission.csv"