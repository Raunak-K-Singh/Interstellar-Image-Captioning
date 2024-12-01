import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    embed_size: int = 256  # Size of word embeddings
    hidden_size: int = 512  # Size of LSTM hidden states
    batch_size: int = 4     # Training batch size
    learning_rate: float = 5e-6  # Learning rate for the optimizer
    num_epochs: int = 10    # Number of training epochs
    max_caption_length: int = 20  # Maximum length of captions
