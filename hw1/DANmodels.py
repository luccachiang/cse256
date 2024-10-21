import torch
import torch.nn as nn
import torch.nn.functional as F
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset
import numpy as np
import os
# import sys
# sys.path.append('./minbpe')

class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, emb, emb_device=None, emb_freeze=True,
                 n_emb=10000, d_emb=300, bpe_encoding=False, bpe_pretrained_path=None, 
                 bpe_vsize=4096, train_emb=None):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile) # [list of words] (sentence), label
        self.emb_freeze = emb_freeze
        self.emb_device = emb_device
        self.bpe_encoding = bpe_encoding
        self.bpe_vsize = bpe_vsize
        # # Extract sentences and lbels from the exaamples
        # self.sentences = [ex.words for ex in self.examples] # a list of wordlists
        # self.labels = [ex.label for ex in self.examples] # a list of labels
        
        # text representation (embedding), move to DAN
        if emb == 'none': # incorporating minbpe here, Important, pass emb here to dev set
            if self.bpe_encoding:
                from minbpe.minbpe import BasicTokenizer
                self.word_indexer = BasicTokenizer()
                # train/load bpe
                corpus_words_list = []
                for ex in self.examples:
                    corpus_words_list.extend(ex.words)
                if os.path.exists(f"bpe_hw1_{bpe_vsize}.model"): # 4096
                    self.word_indexer.load(f"bpe_hw1_{bpe_vsize}.model")
                else:
                    self.word_indexer.train(" ".join(corpus_words_list), vocab_size=self.bpe_vsize)
                    self.word_indexer.save(f"bpe_hw1_{bpe_vsize}")
                if not 'dev' in infile:
                    self.emb = nn.Embedding(self.bpe_vsize, d_emb).to(self.emb_device)
                else:
                    assert train_emb is not None
                    self.emb = train_emb.to(self.emb_device)
            else:
                self.word_indexer = read_word_embeddings("data/glove.6B.300d-relativized.txt").word_indexer # this may not be good
                if not 'dev' in infile:
                    self.emb = nn.Embedding(len(self.word_indexer), d_emb).to(self.emb_device) # maybe just specify a large N number, like allocate an array in c++
                else:
                    assert train_emb is not None
                    self.emb = train_emb.to(self.emb_device)
            # for p in self.emb.parameters():
            #     assert p.requires_grad
        elif emb in 'glove.6B.50d-relativized':
            self.emb = read_word_embeddings("data/glove.6B.50d-relativized.txt") # emb dim 50
            self.emb, self.word_indexer = self.emb.get_initialized_embedding_layer(emb_freeze) # type nn.embedding
            self.emb.to(self.emb_device)
        elif emb in 'glove.6B.300d-relativized':
            self.emb = read_word_embeddings("data/glove.6B.300d-relativized.txt") # emb dim 300
            self.emb, self.word_indexer = self.emb.get_initialized_embedding_layer(emb_freeze)
            self.emb.to(self.emb_device)
        else:
            raise NotImplementedError

    def parameters(self):
        return self.emb.parameters()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        sentence_emb_list = [] # num_words_in_a_sentence OR bpe emb lists, emb dim
        if self.bpe_encoding:
            tokens_of_a_sent = self.word_indexer.encode(" ".join(self.examples[idx].words))
            sentence_emb_list = [self.emb(torch.tensor(token_idx).to(self.emb_device)) for token_idx in tokens_of_a_sent]
            # import ipdb; ipdb.set_trace()
        else:
            for word in self.examples[idx].words:
                # emb_word = self.emb.get_embedding(word)
                word_idx = self.word_indexer.index_of(word) if self.word_indexer.index_of(word) != -1 else self.word_indexer.index_of("UNK")
                word_idx = torch.tensor(word_idx).to(self.emb_device)
                emb_word = self.emb(word_idx)
                sentence_emb_list.append(emb_word)
            
        # import ipdb; ipdb.set_trace()
        return torch.vstack(sentence_emb_list).mean(0), torch.tensor(self.examples[idx].label, dtype=torch.long)
        # try:
        #     return torch.vstack(sentence_emb_list).mean(0), torch.tensor(self.examples[idx].label, dtype=torch.long)
        # except IndexError:
        #     print(idx)                                    
        # return torch.tensor(np.array(sentence_emb_list), dtype=torch.float32).mean(0), torch.tensor(self.examples[idx].label, dtype=torch.long)


class DAN(nn.Module):

    def __init__(self,
                 emb=None,
                 n_embed=10000, # word (index) #
                 d_embed=50, # feature
                 d_hidden=256, # mlp hidden
                 d_out=2,
                 dp=0.2,
                 freeze_emb=True,
                 bn=False,
                 layer3=False):
        super(DAN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layer3 = layer3

        self.dropout1 = nn.Dropout(dp)
        self.bn1 = nn.BatchNorm1d(d_embed) if bn else nn.Identity()
        self.fc1 = nn.Linear(d_embed, d_hidden)
        self.dropout2 = nn.Dropout(dp)
        self.bn2 = nn.BatchNorm1d(d_hidden) if bn else nn.Identity()
        if self.layer3:
            self.fc2 = nn.Linear(d_hidden, d_hidden)
            self.dropout3 = nn.Dropout(dp)
            self.bn3 = nn.BatchNorm1d(d_hidden) if bn else nn.Identity()
            self.fc3 = nn.Linear(d_hidden, d_out)
        else:
            self.fc2 = nn.Linear(d_hidden, d_out)
        self.log_softmax = nn.LogSoftmax(dim=1) # this is important!

    def forward(self, x): # list of word

        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.fc2(x)
        if self.layer3:
            x = self.dropout3(x)
            x = self.bn3(x)
            x = self.fc3(x)
        x = self.log_softmax(x)

        return x