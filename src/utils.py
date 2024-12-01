import json
import re
from PIL import Image
import torchvision.transforms as transforms


def load_vocab(vocab_file):
    """
    Load vocabulary from JSON file.
    """
    with open(vocab_file) as f:
        data = json.load(f)
    return data['vocab']


def transform_image(image_path):
    """
    Transform an image to tensor format suitable for the model.
    """
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    return transform(image)


def tokenize_captions(captions, vocab):
    """Tokenizes and pads captions to a fixed length."""

    tokenized_captions = []

    for caption in captions:
        caption = caption.lower()
        caption = re.sub(r"[^\w\s]", "", caption)

        tokens = caption.split()

        token_indices = [vocab.get(word, vocab['<unk>']) for word in tokens]

        if len(token_indices) < 20:  # Assuming max length is set to 20
            token_indices += [vocab['<pad>']] * (20 - len(token_indices))
        else:
            token_indices = token_indices[:20]

        tokenized_captions.append(token_indices)

    return torch.tensor(tokenized_captions)
