import argparse
import torch
import numpy as np
from tqdm import tqdm
from model import GRUModel, LSTMModel, PackedLSTMModel
from dataset import TextDataset, make_data_loader
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from util import MyCollate

def acc(pred,label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def train(args, data_loader, valid_loader, model):
    """
    TODO: Change the training code as you need. (e.g. different optimizer, different loss function, etc.)
            You can add validation code. -> This will increase the accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    min_loss = np.Inf

    # for visualising learning graph
    train_loss_save, valid_loss_save = [], []
    train_acc_save, valid_acc_save = [], []

    for epoch in range(args.num_epochs):
        train_losses, valid_losses = [], []
        train_acc, valid_acc = 0.0, 0.0
        total_t, total_v = 0, 0
        print("\n")
        print(f"[Epoch {epoch+1} / {args.num_epochs}]")
        
        # Training
        model.train()
        for i, (text, label) in enumerate(tqdm(data_loader)):
            
            ## for not pack padded sequence
            text = text.to(args.device)
            label = label.to(args.device)

            optimizer.zero_grad()

            #output = model(text, input_lengths)
            output, _ = model(text)
            #print("output size: ", output.size()) # output size:  torch.Size([1, 128, 4]) packed 일 때
            #print("output size: ", output.size())  # output size:  torch.Size([128, 4])
            
            # output 모양만 바뀐다. 튜플이다, 여기서는 지금.
            # label은 모델이 달라도 torch.Size([1, 128]) 이다.
            #print("label size: ", label.size())
            label = label.squeeze()
            #print("label size: ", label.size()) # torch.Size([128])
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            total_t += label.size(0)

            train_acc += acc(output, label)

            
        # Validation
        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc/total_t
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.5f}'.format(epoch_train_acc*100))

        model.eval()
        for i, (text, label) in enumerate(tqdm(valid_loader)):
            
            ## for not pack padded sequence
            text = text.to(args.device)
            label = label.to(args.device)            
            optimizer.zero_grad()

            output, _ = model(text)
            
            label = label.squeeze()
            loss = criterion(output, label)

            valid_losses.append(loss.item())
            total_v += label.size(0)

            valid_acc += acc(output, label)

        epoch_valid_loss = np.mean(valid_losses)
        epoch_valid_acc = valid_acc/total_v
        print('valid_accuracy : {:.5f}'.format(epoch_valid_acc*100))



        # Save Model
        if epoch_valid_loss < min_loss:
            torch.save(model.state_dict(), './model/model_temp.pt')
            print('Valid loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(min_loss, epoch_valid_loss))
            min_loss = epoch_valid_loss

        # Save loss and acc
        train_loss_save.append(epoch_train_loss)
        valid_loss_save.append(epoch_valid_loss)
        train_acc_save.append(epoch_train_acc)
        valid_acc_save.append(epoch_valid_acc)
    
    # Save learning graph
    fig = plt.figure(figsize=(16,10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(range(1,len(train_loss_save)+1), train_loss_save, label = 'Training loss')
    ax1.plot(range(1,len(valid_loss_save)+1), valid_loss_save, label = 'Validation loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('loss graph', fontsize=15)
    plt.legend(loc='lower left', ncol=2)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(range(1,len(train_acc_save)+1), train_acc_save, label = 'Training acc')
    ax2.plot(range(1,len(valid_acc_save)+1), valid_acc_save, label = 'Validation acc')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.set_title('accuracy graph', fontsize=15)
    
    plt.legend(loc='lower left', ncol=2)
    plt.tight_layout()
    plt.show()
    fig.savefig('./img/model_temp.png', bbox_inches = 'tight')


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='2022 DL Term Project #2')
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training (default: 64)")
    parser.add_argument('--vocab_size', type=int, default=30000, help="maximum vocab size")
    parser.add_argument('--batch_first', action='store_true', help="If true, then the model returns the batch first")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs to train for (default: 5)")
    
    args = parser.parse_args()

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
    train(args, train_loader, valid_loader, model)
