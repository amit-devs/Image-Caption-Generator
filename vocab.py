import re
from collections import Counter


class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold

        # Special tokens
        self.itos = {
            0: "<PAD>",
            1: "<START>",
            2: "<END>",
            3: "<UNK>"
        }

        self.stoi = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2,
            "<UNK>": 3
        }

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        text = text.lower()

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Keep words only
        tokens = re.findall(r"\b\w+\b", text)

        return tokens

    def build_vocabulary(self, captions):
        frequencies = Counter()

        # Count words
        for caption in captions:
            tokens = self.tokenizer(caption)

            for token in tokens:
                frequencies[token] += 1

        # Add words that occur frequently enough
        idx = len(self.itos)

        for word, frequency in frequencies.items():

            if frequency >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokens = self.tokenizer(text)

        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokens
        ]