import gc
import os
import warnings

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from configs.config import CFG, ENSEMBLE_META_PATH, FINAL_MODEL_PATH
from src.dataset import MultiViewDataset
from src.model import MultiViewClassifier
from src.transforms import get_stage2_train_transform
from src.utils import seed_everything, label_to_int, load_json

warnings.filterwarnings("ignore")


def build_loader():
    base_path = CFG["BASE_PATH"]
    dev_df = pd.read_csv(os.path.join(base_path, "dev.csv"))
    dev_df["label"] = dev_df["label"].apply(label_to_int)

    dataset = MultiViewDataset(
        df=dev_df,
        image_root=os.path.join(base_path, "dev"),
        transform=get_stage2_train_transform(CFG["IMG_SIZE"]),
        is_test=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=CFG["STAGE2_FINE_BATCH_SIZE"],
        shuffle=True,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
    )
    return loader


def main():
    if not os.path.exists(ENSEMBLE_META_PATH):
        raise FileNotFoundError("먼저 train_stagewise.py를 실행해서 fold checkpoint를 만들어야 합니다.")

    seed_everything(CFG["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_json(ENSEMBLE_META_PATH)
    best_fold = sorted(meta["folds"], key=lambda x: x["best_logloss"])[0]

    model = MultiViewClassifier(model_name=CFG["MODEL_NAME"], dropout=CFG["DROPOUT"]).to(device)
    bundle = torch.load(best_fold["finetune_ckpt"], map_location=device)
    model.load_state_dict(bundle["model_state_dict"])

    loader = build_loader()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["STAGE2_FINE_LR"],
        weight_decay=CFG["STAGE2_FINE_WEIGHT_DECAY"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=CFG["AMP"])

    refit_epochs = max(2, CFG["STAGE2_FINE_EPOCHS"])
    model.train()
    for epoch in range(1, refit_epochs + 1):
        losses = []
        for views, labels in tqdm(loader, desc=f"Refit Epoch {epoch}", leave=False):
            views = [v.to(device) for v in views]
            labels = labels.unsqueeze(1).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=CFG["AMP"]):
                logits = model(views)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        print(f"[Refit Epoch {epoch}/{refit_epochs}] Loss: {sum(losses)/len(losses):.4f}")

    torch.save({"model_state_dict": model.state_dict(), "temperature": best_fold.get("temperature", 1.0)}, FINAL_MODEL_PATH)
    print(f"Final refit model saved -> {FINAL_MODEL_PATH}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
