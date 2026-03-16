from pathlib import Path

CFG = {
    # -----------------------------
    # 경로
    # -----------------------------
    "BASE_PATH": "/content/drive/MyDrive/데이콘/open",
    "OUTPUT_DIR": "./outputs",

    # -----------------------------
    # 재현성 / 시스템
    # -----------------------------
    "SEED": 42,
    "NUM_WORKERS": 2,
    "AMP": True,
    "DEVICE": "cuda",

    # -----------------------------
    # 모델
    # -----------------------------
    "MODEL_NAME": "efficientnet_b3",
    "IMG_SIZE": 384,
    "DROPOUT": 0.35,

    # -----------------------------
    # Stage1: official rule 유지
    # train -> dev 로 pretrain
    # -----------------------------
    "STAGE1_EPOCHS": 10,
    "STAGE1_BATCH_SIZE": 8,
    "STAGE1_LR": 5e-5,
    "STAGE1_WEIGHT_DECAY": 2e-4,
    "STAGE1_PATIENCE": 3,

    # -----------------------------
    # Stage2: dev 내부 KFold adaptation
    # dev만 사용, test 사용 안 함
    # -----------------------------
    "STAGE2_N_SPLITS": 5,
    "STAGE2_HEAD_EPOCHS": 3,
    "STAGE2_HEAD_BATCH_SIZE": 8,
    "STAGE2_HEAD_LR": 1.5e-4,
    "STAGE2_HEAD_WEIGHT_DECAY": 1e-4,
    "STAGE2_HEAD_PATIENCE": 2,

    "STAGE2_FINE_EPOCHS": 3,
    "STAGE2_FINE_BATCH_SIZE": 8,
    "STAGE2_FINE_LR": 2e-6,
    "STAGE2_FINE_WEIGHT_DECAY": 1e-4,
    "STAGE2_FINE_PATIENCE": 2,
    "UNFREEZE_LAST_N_BLOCKS": 1,

    # -----------------------------
    # Inference
    # -----------------------------
    "TTA": True,
    "CLIP_MIN": 1e-6,
    "CLIP_MAX": 1 - 1e-6,
    "USE_TEMPERATURE_SCALING": True,

    # 제출 컬럼명 자동 매핑 실패 시 사용
    "POSITIVE_LABEL_COL": "unstable_prob",
    "NEGATIVE_LABEL_COL": "stable_prob",
}

OUTPUT_DIR = Path(CFG["OUTPUT_DIR"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRETRAIN_MODEL_PATH = str(OUTPUT_DIR / "stage1_pretrain_best.pth")
STAGE2_DIR = OUTPUT_DIR / "stage2_folds"
STAGE2_DIR.mkdir(parents=True, exist_ok=True)
ENSEMBLE_META_PATH = str(STAGE2_DIR / "ensemble_meta.json")
FINAL_MODEL_PATH = str(OUTPUT_DIR / "final_model_refit_dev_all.pth")
SUBMISSION_PATH = str(OUTPUT_DIR / "submission.csv")
