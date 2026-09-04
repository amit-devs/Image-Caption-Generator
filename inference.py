import pickle

import torch
from PIL import Image
from torchvision import transforms

from model import CNNEncoder, DecoderWithAttention


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ENCODER_DIM = 256
EMBED_SIZE = 256
DECODER_DIM = 512
ATTENTION_DIM = 256


# ============================================================
# LOAD VOCABULARY
# ============================================================

with open(
    "models/vocab.pkl",
    "rb"
) as file:

    vocab = pickle.load(file)


# ============================================================
# CREATE MODELS
# ============================================================

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
# LOAD TRAINED WEIGHTS
# ============================================================

encoder.load_state_dict(
    torch.load(
        "models/encoder.pth",
        map_location=DEVICE
    )
)

decoder.load_state_dict(
    torch.load(
        "models/decoder.pth",
        map_location=DEVICE
    )
)

encoder.eval()
decoder.eval()


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# GENERATE CAPTION
# ============================================================

def generate_caption(image_path, max_length=30):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        encoder_out = encoder(image)

        h, c = decoder.init_hidden_state(
            encoder_out
        )

        word = vocab.stoi["<START>"]

        caption = []

        for _ in range(max_length):

            word_tensor = torch.tensor(
                [word],
                device=DEVICE
            )

            embedding = decoder.embedding(
                word_tensor
            )

            attention_encoding, _ = decoder.attention(
                encoder_out,
                h
            )

            lstm_input = torch.cat(
                [
                    embedding,
                    attention_encoding
                ],
                dim=1
            )

            h, c = decoder.lstm(
                lstm_input,
                (h, c)
            )

            output = decoder.fc(h)

            predicted_word = output.argmax(
                dim=1
            ).item()

            if predicted_word == vocab.stoi["<END>"]:
                break

            if predicted_word != vocab.stoi["<PAD>"]:

                caption.append(
                    vocab.itos[predicted_word]
                )

            word = predicted_word

    return " ".join(caption)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    image_path = input(
        "Enter image path: "
    )

    caption = generate_caption(
        image_path
    )

    print("\nGenerated Caption:")
    print(caption)