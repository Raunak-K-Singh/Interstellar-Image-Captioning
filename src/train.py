import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from config import ModelConfig  # Import configuration settings from config.py.
from models import ImageCaptioningModel  # Import your image captioning model class.
from utils import tokenize_captions  # Import utility functions


class CaptionDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        with open(annotations_file) as f:
            self.captions_data = json.load(f)

        self.img_dir = img_dir

        self.vocab = {word: idx for idx, word in enumerate(self.captions_data['vocab'])}
        self.vocab['<pad>'] = 0  # Padding token index.
        self.vocab['<unk>'] = len(self.vocab)  # Unknown token index.

        self.max_caption_length = 20

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.captions_data['annotations'])

    def __getitem__(self, idx):
        img_id = self.captions_data['annotations'][idx]['image_id']

        img_path = os.path.join(self.img_dir, img_id)

        image = Image.open(img_path).convert('RGB')

        image_tensor = self.transform(image)

        captions = [caption['caption'] for caption in self.captions_data['annotations'] if
                    caption['image_id'] == img_id]

        caption_tensor = tokenize_captions(captions, self.vocab)

        return image_tensor, caption_tensor


def train_model():
    config = ModelConfig()

    dataset = CaptionDataset('data/captions.json', 'data/train2017')
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    vocab_size = len(dataset.captions_data['vocab'])

    model = ImageCaptioningModel(vocab_size=vocab_size)
    model.to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        total_loss = 0

        model.train()

        for images, captions in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}"):
            optimizer.zero_grad()

            images = images.to(config.device)
            captions_tensor = captions.to(config.device)

            outputs = model(images, captions_tensor)

            loss = criterion(outputs.view(-1, outputs.size(2)), captions_tensor.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{config.num_epochs}], Loss: {total_loss / len(data_loader):.4f}')

    torch.save(model.state_dict(), 'models/model.pth')
    print("Model saved to 'models/model.pth'")


if __name__ == "__main__":
    train_model()
