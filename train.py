import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import MultiViewDataset
from src.model import MultiViewResNet
from configs.config import CFG, BASE_PATH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_df = pd.read_csv(f"{BASE_PATH}/train.csv")
val_df = pd.read_csv(f"{BASE_PATH}/dev.csv")


train_transform = transforms.Compose([
    transforms.Resize((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
    transforms.ToTensor(),
])


train_dataset = MultiViewDataset(train_df, f"{BASE_PATH}/train", train_transform)
val_dataset = MultiViewDataset(val_df, f"{BASE_PATH}/dev", test_transform)


train_loader = DataLoader(train_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=False)


model = MultiViewResNet().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=CFG["LEARNING_RATE"])


for epoch in range(CFG["EPOCHS"]):

    model.train()

    for views, labels in train_loader:

        views = [v.to(device) for v in views]
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(views)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1} finished")
