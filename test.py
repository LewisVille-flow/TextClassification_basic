import argparse
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import GRUModel, LSTMModel
from dataset import TextDataset, make_data_loader
from sklearn.metrics import classification_report


def test(args, data_loader, model, embedding_dim):
    true = np.array([])
    pred = np.array([])
    model.eval()
    for i, (text, label) in enumerate(tqdm(data_loader)):

        text = text.to(args.device)
        label = label.to(args.device)            

        output, _ = model(text)
        
        label = label.squeeze()
        output = output.argmax(dim=-1)
        output = output.detach().cpu().numpy()

        pred = np.append(pred, output)
        
        label = label.detach().cpu().numpy()
        true =  np.append(true, label, axis=0)

    return pred, true


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2022 DL Term Project #2')
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training (default: 64)")
    parser.add_argument('--vocab_size', type=int, default=30000, help="maximum vocab size")
    parser.add_argument('--batch_first', action='store_true', help="If true, then the model returns the batch first")

    args = parser.parse_args()

    """
    TODO: You MUST write the same model parameters as in the train.py file !!
    """
    # Model parameters
    input_size = args.vocab_size
    output_size = 4     # num of classes
    embedding_dim = 100 # embedding dimension
    hidden_dim = 64  # hidden size of RNN
    num_layers = 1
        

    # Make Test Loader
    test_dataset = TextDataset(args.data_dir, 'test', args.vocab_size)
    args.pad_idx = test_dataset.sentences_vocab.wtoi['<PAD>']
    test_loader = make_data_loader(test_dataset, args.batch_size, args.batch_first, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # instantiate model
    model = LSTMModel(input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first=args.batch_first)
    model.load_state_dict(torch.load('./model/model_20230209_2200_LSTM.pt'))
    model = model.to(device)
    
    print(test_dataset.labels_vocab.itow)
    target_names = [ w for i, w in test_dataset.labels_vocab.itow.items()]
    # Test The Model
    pred, true = test(args, test_loader, model, embedding_dim)
    print(pred)
    print(true)
    
    accuracy = (true == pred).sum() / len(pred)
    print("Test Accuracy : {:.5f}".format(accuracy))

    ## Save result
    strFormat = '%12s%12s\n'

    with open('result.txt', 'w') as f:
        f.write('Test Accuracy : {:.5f}\n'.format(accuracy))
        f.write('true label  |  predict label \n')
        f.write('-------------------------- \n')
        
        for i in range(len(pred)):
            f.write(strFormat % (test_dataset.labels_vocab.itow[true[i]],test_dataset.labels_vocab.itow[pred[i]]))
            
  
    # print(classification_report(true, pred, target_names=target_names))
    