train_loader = MultiViewDataset(
    train_df,
    image_root=os.path.join(BASE_PATH, "train"),
)

valid_loader = MultiViewDataset(
    dev_df,
    image_root=os.path.join(BASE_PATH, "dev"),
)