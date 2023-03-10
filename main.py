import argparse
import torch
import numpy as np
from tqdm import tqdm
from model import GRUModel, LSTMModel, PackedLSTMModel
from dataset import TextDataset, make_data_loader
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from util import MyCollate, read_json
from train import train


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2022 DL Term Project #2')
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training (default: 64)")
    parser.add_argument('--vocab_size', type=int, default=30000, help="maximum vocab size")
    parser.add_argument('--batch_first', action='store_true', help="If true, then the model returns the batch first")
    #parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs to train for (default: 5)")
    
    args = parser.parse_args()

    # config test
    fname = Path("config.json")
    config = read_json(fname)

    """
    TODO: Build your model Parameters. You can change the model architecture and hyperparameters as you wish.
            (e.g. change epochs, vocab_size, hidden_dim etc.)
    """
    # Model hyperparameters
    input_size = args.vocab_size
    output_size = 4     # num of classes
    embedding_dim = 100 # embedding dimension
    hidden_dim = 64  # hidden size of RNN
    num_layers = 1
    
    # Make Train Loader
    train_dataset = TextDataset(args.data_dir, 'train', args.vocab_size)
    args.pad_idx = train_dataset.sentences_vocab.wtoi['<PAD>']
    train_loader = make_data_loader(train_dataset, args.batch_size, args.batch_first, shuffle=True)

    # Valid Loader(added)
    valid_dataset = TextDataset(args.data_dir, 'val', args.vocab_size)
    args.pad_idx = valid_dataset.sentences_vocab.wtoi['<PAD>']
    valid_loader = make_data_loader(valid_dataset, args.batch_size, args.batch_first, shuffle=True)
    

    # train val split test
    print("train and valid length each: {}, {}".format(len(train_dataset.sentences), len(valid_dataset.sentences)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print("device : ", device)

    # instantiate model
    #model = BaseModel(input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first=args.batch_first)
    model = LSTMModel(input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first=args.batch_first)
    #model = PackedLSTMModel(input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first=args.batch_first)
    
    model = model.to(device)

    # Training The Model
    print("##### start training #####\n")
    train(args, config, train_loader, valid_loader, model)