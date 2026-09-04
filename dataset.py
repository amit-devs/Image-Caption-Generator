import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FlickrDataset(Dataset):

    def __init__(
        self,
        root_dir="dataset/Images",
        caption_file="dataset/captions.txt",
        vocab=None,
        transform=None
    ):
        self.root_dir = root_dir
        self.vocab = vocab

        # Image transformations
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Read captions
        self.data = pd.read_csv(caption_file)

        # Make sure the column names are correct
        self.data.columns = ["image", "caption"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_name = row["image"]
        caption = row["caption"]

        # Load image
        image_path = os.path.join(
            self.root_dir,
            image_name
        )

        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        image = self.transform(image)

        # Convert caption into numerical IDs
        caption_ids = []

        if self.vocab is not None:

            caption_ids.append(
                self.vocab.stoi["<START>"]
            )

            caption_ids.extend(
                self.vocab.numericalize(caption)
            )

            caption_ids.append(
                self.vocab.stoi["<END>"]
            )

        return image, torch.tensor(
            caption_ids,
            dtype=torch.long
        )