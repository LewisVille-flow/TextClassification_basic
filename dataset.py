import pandas as pd
import os
import torch

from torch.utils.data import Dataset, DataLoader
from vocab import Vocabulary
from util import MyCollate

import torchtext.vocab as vocab

def make_data_loader(dataset, batch_size, batch_first, shuffle=True): #increase num_workers according to CPU
    #get pad_idx for collate fn
    pad_idx = dataset.sentences_vocab.wtoi['<PAD>']
    #define loader
    loader = DataLoader(dataset, batch_size = batch_size, shuffle=shuffle,
                        collate_fn = MyCollate(pad_idx=pad_idx, batch_first=batch_first)) #MyCollate class runs __call__ method by default
    return loader

class TextDataset(Dataset):

    def __init__ (self, data_dir, mode, vocab_size):
        if mode == 'val':
            self.df = pd.read_csv(os.path.join(data_dir, 'train' + '.csv'))
        else:
            self.df = pd.read_csv(os.path.join(data_dir, mode + '.csv'))

        self.sentences = self.df['text'].values
        self.labels = self.df['label'].values

        if mode == 'val' or mode == 'train':
            self.sentences_train, self.sentences_val = torch.utils.data.random_split(self.sentences, [120000, 30000], generator=torch.Generator().manual_seed(42))
            self.labels_train, self.labels_val = torch.utils.data.random_split(self.labels, [120000, 30000], generator=torch.Generator().manual_seed(42))

            if mode == 'train':
                self.sentences = self.sentences_train
                self.labels = self.labels_train
            elif mode == 'val':
                self.sentences = self.sentences_val  
                self.labels = self.labels_val


        # Initialize dataset Vocabulary object and build our vocabulary
        self.sentences_vocab = Vocabulary(vocab_size)
        self.labels_vocab = Vocabulary(vocab_size)


        """
        Build or Load Vocabulary

        """
        if mode == 'train': 
            self.sentences_vocab.build_vocabulary(self.sentences)
            self.labels_vocab.build_vocabulary(self.labels, add_unk=False)

            self.sentences_vocab.save_vocabulary('data')
            self.labels_vocab.save_vocabulary('label')
        
        elif mode == 'test' or mode == 'val':
            self.sentences_vocab.load_vocabulary('data', './pickles')
            self.labels_vocab.load_vocabulary('label', './pickles')

    def __len__(self):
        return len(self.labels)

    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        #numericalize the sentence ex) ['cat', 'in', 'a', 'bag'] -> [2,3,9,24,22]
        numeric_sentence = self.sentences_vocab.sentence_to_numeric(sentence)
        numeric_label = self.labels_vocab.sentence_to_numeric(label)

        # return inputs, label
        return torch.tensor(numeric_sentence), torch.tensor(numeric_label)