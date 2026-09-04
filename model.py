import torch
import torch.nn as nn
import torchvision.models as models


# ============================================================
# CNN ENCODER
# ============================================================

class CNNEncoder(nn.Module):
    def __init__(self, encoder_dim=256):
        super(CNNEncoder, self).__init__()

        # Load pre-trained ResNet50
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        # Remove the average pooling and classification layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # ResNet50 produces 2048-dimensional features.
        # Convert them to our encoder dimension.
        self.linear = nn.Linear(2048, encoder_dim)

    def forward(self, images):
        # ResNet output:
        # (batch_size, 2048, 7, 7)

        features = self.resnet(images)

        # Change:
        # (batch_size, 2048, 7, 7)
        #
        # Into:
        # (batch_size, 49, 2048)

        features = features.permute(0, 2, 3, 1)

        features = features.view(
            features.size(0),
            -1,
            features.size(-1)
        )

        # Convert 2048 features to encoder_dim
        features = self.linear(features)

        # Output:
        # (batch_size, 49, encoder_dim)

        return features


# ============================================================
# ATTENTION
# ============================================================

class Attention(nn.Module):
    def __init__(
        self,
        encoder_dim,
        decoder_dim,
        attention_dim
    ):
        super(Attention, self).__init__()

        # Transform CNN features
        self.encoder_att = nn.Linear(
            encoder_dim,
            attention_dim
        )

        # Transform LSTM hidden state
        self.decoder_att = nn.Linear(
            decoder_dim,
            attention_dim
        )

        # Produce one attention score
        self.full_att = nn.Linear(
            attention_dim,
            1
        )

        self.relu = nn.ReLU()

        # Convert scores into probabilities
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        encoder_out,
        decoder_hidden
    ):
        """
        encoder_out:
            (batch_size, num_pixels, encoder_dim)

        decoder_hidden:
            (batch_size, decoder_dim)
        """

        # Process image features
        att1 = self.encoder_att(encoder_out)

        # Process LSTM hidden state
        att2 = self.decoder_att(decoder_hidden)

        # Combine image and decoder information
        att = self.full_att(
            self.relu(
                att1 + att2.unsqueeze(1)
            )
        ).squeeze(2)

        # Attention weights
        alpha = self.softmax(att)

        # Weighted image features
        attention_weighted_encoding = (
            encoder_out *
            alpha.unsqueeze(2)
        ).sum(dim=1)

        return attention_weighted_encoding, alpha


# ============================================================
# LSTM DECODER WITH ATTENTION
# ============================================================

class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        embed_size,
        decoder_dim,
        vocab_size,
        encoder_dim=256,
        attention_dim=256
    ):
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        # Attention mechanism
        self.attention = Attention(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim
        )

        # Convert word IDs into word embeddings
        self.embedding = nn.Embedding(
            vocab_size,
            embed_size
        )

        # LSTM cell
        #
        # Input:
        # word embedding + attended image features
        #
        self.lstm = nn.LSTMCell(
            embed_size + encoder_dim,
            decoder_dim
        )

        # Initial hidden state
        self.init_h = nn.Linear(
            encoder_dim,
            decoder_dim
        )

        # Initial cell state
        self.init_c = nn.Linear(
            encoder_dim,
            decoder_dim
        )

        # Convert LSTM hidden state into vocabulary scores
        self.fc = nn.Linear(
            decoder_dim,
            vocab_size
        )

        self.dropout = nn.Dropout(0.5)

    # --------------------------------------------------------
    # Initialize LSTM hidden and cell states
    # --------------------------------------------------------

    def init_hidden_state(self, encoder_out):

        # Average all image regions
        mean_encoder = encoder_out.mean(dim=1)

        h = self.init_h(mean_encoder)

        c = self.init_c(mean_encoder)

        return h, c

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    def forward(
        self,
        encoder_out,
        captions
    ):
        """
        encoder_out:
            (batch_size, num_pixels, encoder_dim)

        captions:
            (batch_size, max_caption_length)
        """

        batch_size = encoder_out.size(0)

        # Initialize hidden and cell states
        h, c = self.init_hidden_state(
            encoder_out
        )

        # Convert caption word IDs into embeddings
        embeddings = self.embedding(captions)

        sequence_length = captions.size(1)

        # Store predictions
        predictions = torch.zeros(
            batch_size,
            sequence_length,
            self.vocab_size,
            device=encoder_out.device
        )

        # Generate one word at a time
        for t in range(sequence_length):

            # Attention focuses on relevant image regions
            attention_weighted_encoding, alpha = self.attention(
                encoder_out,
                h
            )

            # Combine:
            # current word embedding
            # +
            # attended image features
            lstm_input = torch.cat(
                [
                    embeddings[:, t, :],
                    attention_weighted_encoding
                ],
                dim=1
            )

            # LSTM step
            h, c = self.lstm(
                lstm_input,
                (h, c)
            )

            # Predict next word
            preds = self.fc(
                self.dropout(h)
            )

            predictions[:, t, :] = preds

        return predictions