# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer # pip install scikit-learn matplotlib
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import DAN, SentimentDatasetDAN
import csv

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer, args):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train() # TODO maybe dataset emb set to training mode?, same with eval
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float().to(args.device)
        y = y.to(args.device)

        # Compute prediction error
        pred = model(X)
        # import ipdb; ipdb.set_trace()
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer, args):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float().to(args.device)
        y = y.to(args.device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader, args):
    model.to(args.device)
    loss_fn = nn.NLLLoss()
    # import ipdb; ipdb.set_trace()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(train_loader.dataset.parameters()), lr=0.0001) # TODO add params of embedding

    all_train_accuracy = []
    all_test_accuracy = []
    all_train_loss = []
    all_test_loss = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer, args)
        all_train_accuracy.append(train_accuracy)
        all_train_loss.append(train_loss)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer, args)
        all_test_accuracy.append(test_accuracy)
        all_test_loss.append(test_loss)

        if epoch % 5 == 4:
            print(f'Epoch #{epoch + 1}: train loss {train_loss:.3f}, train accuracy {train_accuracy:.3f}, dev loss {test_loss:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy, all_train_loss, all_test_loss


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')
    parser.add_argument('--emb', type=str, required=False, default='none', help='Which emb to use')
    parser.add_argument('--emb_freeze', type=bool, required=False, default=True, help='Whether to freeze pretrained emb')
    parser.add_argument('--layer3', type=bool, required=False, default=False, help='2-layer or 3-layer ffn')
    parser.add_argument('--bpe_encoding', type=bool, required=False, default=False, help='Whether to use bpe encoding when randomly initialing emb')
    parser.add_argument('--bpe_vocab_size', type=int, required=False, default=4096, help='BPE vocabulary size')
    parser.add_argument('--dp', type=float, required=False, default=0.0, help='Dropout rate')
    parser.add_argument('--lr', type=float, required=False, default=0.0001, help='learning rate')
    parser.add_argument('--hiddensize', type=int, required=False, default=256, help='hidden layer size of ffn')
    # parser.add_argument('--bpe_pretrained_path', type=str, required=False, default=None, help='cpu, cuda:0, etc.')
    parser.add_argument('--device', type=str, required=False, default='cuda:4', help='cpu, cuda:0, etc.')
    parser.add_argument('--comment', type=str, required=False, default='test', help='name for the run')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load dataset
    start_time = time.time()

    if args.model == 'BOW':
        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt")
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)
    elif args.model in ['DAN', 'SUBWORDDAN']:
        if args.model == 'SUBWORDDAN':
            args.emb = 'none'
            args.bpe_encoding = True
        train_data = SentimentDatasetDAN("data/train.txt", emb=args.emb, emb_freeze=True, emb_device=args.device,
                                             bpe_encoding=args.bpe_encoding, bpe_vsize=args.bpe_vocab_size, train_emb=None)
        dev_data = SentimentDatasetDAN("data/dev.txt", emb=args.emb, emb_freeze=True, emb_device=args.device,
                                             bpe_encoding=args.bpe_encoding, bpe_vsize=args.bpe_vocab_size, train_emb=train_data.emb)
        if args.emb in 'glove.6B.50d-relativized' or args.emb in 'glove.6B.300d-relativized': # pretrain
            train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
            test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)
        else: # random init
            train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
            test_loader = DataLoader(dev_data, batch_size=256, shuffle=False)
    else:
        raise NotImplementedError

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")

    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy, _, _ = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader, args)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy, _, _ = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader, args)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = f'train_accuracy_{args.comment}.pdf'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = f'dev_accuracy_{args.comment}.pdf'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model in ['DAN', 'SUBWORDDAN']:
        #TODO:  Train and evaluate your DAN
        start_time = time.time()
        if args.emb in 'glove.6B.50d-relativized':
            dan_train_accuracy, dan_test_accuracy, dan_train_loss, dan_test_loss = experiment(
                DAN(d_embed=50, d_hidden=args.hiddensize, dp=args.dp, layer3=args.layer3), train_loader, test_loader, args) # hid 512
        elif args.emb in 'glove.6B.300d-relativized':
            dan_train_accuracy, dan_test_accuracy, dan_train_loss, dan_test_loss = experiment(
                DAN(d_embed=300, d_hidden=args.hiddensize, dp=args.dp, layer3=args.layer3), train_loader, test_loader, args) # hid 256
        else: # random init
            dan_train_accuracy, dan_test_accuracy, dan_train_loss, dan_test_loss = experiment(
                DAN(d_embed=300, d_hidden=args.hiddensize, dp=args.dp, bn=True, layer3=args.layer3), train_loader, test_loader, args) # dp 0.2, hid 128
        
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='DAN_acc')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Train Accuracy for DAN')
        plt.legend()
        plt.grid()
        # Save the training accuracy figure
        training_accuracy_file = f'{args.model}_train_acc_{args.comment}.pdf'
        plt.savefig(training_accuracy_file)
        with open(f'{args.model}_train_acc_{args.comment}.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(dan_train_accuracy)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_loss, label='DAN_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Training Loss')
        plt.title('Train Loss for DAN')
        plt.legend()
        plt.grid()
        # Save the training accuracy figure
        training_loss_file = f'{args.model}_train_loss_{args.comment}.pdf'
        plt.savefig(training_loss_file)
        with open(f'{args.model}_train_loss_{args.comment}.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(dan_train_accuracy)
        print(f"\n\nTraining loss plot saved as {training_loss_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_accuracy, label='DAN_acc')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN')
        plt.legend()
        plt.grid()
        # Save the testing accuracy figure
        testing_accuracy_file = f'{args.model}_dev_acc_{args.comment}.pdf'
        plt.savefig(testing_accuracy_file)
        with open(f'{args.model}_dev_acc_{args.comment}.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(dan_train_accuracy)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        plt.figure(figsize=(8, 6))
        plt.plot(dan_test_loss, label='DAN_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Loss')
        plt.title('Dev Loss for DAN')
        plt.legend()
        plt.grid()
        # Save the training accuracy figure
        test_loss_file = f'{args.model}_dev_loss_{args.comment}.pdf'
        plt.savefig(test_loss_file)
        with open(f'{args.model}_dev_loss_{args.comment}.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(dan_train_accuracy)
        print(f"\n\nTraining loss plot saved as {test_loss_file}")

if __name__ == "__main__":
    # python main.py --model DAN --emb none
    main()
