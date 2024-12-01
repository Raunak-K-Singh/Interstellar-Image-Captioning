import torch
import torch.nn as nn
import torchvision.models as models


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, features, hidden_state):
        scores = self.Va(torch.tanh(self.Wa(features) + self.Ua(hidden_state)))
        weights = torch.softmax(scores, dim=1)
        context = (weights * features).sum(dim=1)  # Weighted sum of features
        return context, weights


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512):
        super(ImageCaptioningModel, self).__init__()

        self.encoder = models.resnet50(pretrained=True)
        self.encoder.fc = nn.Identity()  # Remove the final layer

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + 2048, hidden_size)  # ResNet output size is 2048.
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)  # Extract features from images.
        embeddings = self.embedding(captions)  # Get embeddings for words.

        h_t = torch.zeros(embeddings.size(0), self.lstm.hidden_size).to(images.device)
        c_t = torch.zeros(embeddings.size(0), self.lstm.hidden_size).to(images.device)

        outputs = []

        for t in range(captions.size(1)):
            if t == 0:  # For the first time step
                context_vector = features.unsqueeze(1)  # Use image features as context
            else:
                context_vector, _ = self.attention(features.unsqueeze(1), h_t)

            lstm_input = torch.cat((embeddings[:, t], context_vector.squeeze(1)), dim=1)
            h_t, c_t = self.lstm(lstm_input.unsqueeze(1), (h_t, c_t))
            output = self.fc(h_t.squeeze(1))
            outputs.append(output)

        return torch.stack(outputs, dim=1)  # Shape: (batch_size, seq_length, vocab_size)
