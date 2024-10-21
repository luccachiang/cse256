# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset


# Dataset class for handling sentiment analysis data
class SentimentDatasetBOW(Dataset):
    def __init__(self, infile, vectorizer=None, train=True):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile) # [list of words], label
        # import ipdb; ipdb.set_trace()
        
        # Extract sentences and lbels from the exaamples
        self.sentences = [" ".join(ex.words) for ex in self.examples] # a list of sentences
        self.labels = [ex.label for ex in self.examples] # a list of labels
        
        # Vectorize the sentences using CountVectorizer
        if train or vectorizer is None:
            self.vectorizer = CountVectorizer(max_features=512) # bag of words
            self.embeddings = self.vectorizer.fit_transform(self.sentences).toarray() # 6920,512, one sentence to a bow representation of a sentence
        else:
            self.vectorizer = vectorizer
            self.embeddings = self.vectorizer.transform(self.sentences).toarray()
        
        # Convert embeddings and labels to PyTorch tensors
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return self.embeddings[idx], self.labels[idx]


# Two-layer fully connected neural network
class NN2BOW(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x): # does not consider batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

    
# Three-layer fully connected neural network
class NN3BOW(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.log_softmax(x)

