import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from vocab import Vocabulary
from dataset import FlickrDataset
from model import CNNEncoder, DecoderWithAttention


# ============================================================
# SETTINGS
# ============================================================

CAPTION_FILE = "dataset/captions.txt"
IMAGE_DIR = "dataset/Images"

BATCH_SIZE = 32
EMBED_SIZE = 256
ENCODER_DIM = 256
DECODER_DIM = 512
ATTENTION_DIM = 256

NUM_EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# BUILD VOCABULARY
# ============================================================

print("\nBuilding vocabulary...")

captions = []

with open(
    CAPTION_FILE,
    "r",
    encoding="utf-8"
) as file:

    next(file)  # Skip header

    for line in file:

        parts = line.strip().split(",", 1)

        if len(parts) == 2:
            captions.append(parts[1])


vocab = Vocabulary(freq_threshold=5)

vocab.build_vocabulary(captions)

print("Vocabulary size:", len(vocab))


# ============================================================
# SAVE VOCABULARY
# ============================================================

os.makedirs("models", exist_ok=True)

with open(
    "models/vocab.pkl",
    "wb"
) as file:

    pickle.dump(vocab, file)

print("Vocabulary saved.")


# ============================================================
# CREATE DATASET
# ============================================================

print("\nLoading dataset...")

dataset = FlickrDataset(
    root_dir=IMAGE_DIR,
    caption_file=CAPTION_FILE,
    vocab=vocab
)

print("Total samples:", len(dataset))


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    images = []
    captions = []

    for image, caption in batch:

        images.append(image)
        captions.append(caption)

    images = torch.stack(images)

    captions = nn.utils.rnn.pad_sequence(
        captions,
        batch_first=True,
        padding_value=vocab.stoi["<PAD>"]
    )

    return images, captions


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


# ============================================================
# CREATE MODELS
# ============================================================

print("\nCreating models...")

encoder = CNNEncoder(
    encoder_dim=ENCODER_DIM
).to(DEVICE)

decoder = DecoderWithAttention(
    embed_size=EMBED_SIZE,
    decoder_dim=DECODER_DIM,
    vocab_size=len(vocab),
    encoder_dim=ENCODER_DIM,
    attention_dim=ATTENTION_DIM
).to(DEVICE)


# ============================================================
# FREEZE RESNET50
# ============================================================

for parameter in encoder.resnet.parameters():
    parameter.requires_grad = False


# ============================================================
# OPTIMIZERS
# ============================================================

encoder_optimizer = torch.optim.Adam(
    encoder.linear.parameters(),
    lr=LEARNING_RATE
)

decoder_optimizer = torch.optim.Adam(
    decoder.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=vocab.stoi["<PAD>"]
)


# ============================================================
# TRAINING
# ============================================================

print("\nStarting training...\n")

best_loss = float("inf")

for epoch in range(NUM_EPOCHS):

    encoder.train()
    decoder.train()

    total_loss = 0

    for batch_index, (images, captions_batch) in enumerate(
        train_loader
    ):

        images = images.to(DEVICE)
        captions_batch = captions_batch.to(DEVICE)

        # CNN encoder
        encoder_out = encoder(images)

        # LSTM decoder
        predictions = decoder(
            encoder_out,
            captions_batch[:, :-1]
        )

        # Target words
        targets = captions_batch[:, 1:]

        # Flatten
        predictions = predictions.reshape(
            -1,
            predictions.size(-1)
        )

        targets = targets.reshape(-1)

        # Loss
        loss = criterion(
            predictions,
            targets
        )

        # Clear gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

        if batch_index % 100 == 0:

            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                f"Batch {batch_index}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_loader)

    print(
        f"\nEpoch {epoch + 1} "
        f"Average Loss: {average_loss:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if average_loss < best_loss:

        best_loss = average_loss

        torch.save(
            encoder.state_dict(),
            "models/encoder.pth"
        )

        torch.save(
            decoder.state_dict(),
            "models/decoder.pth"
        )

        print("Best model saved!")


print("\n================================")
print("Training completed!")
print("================================")