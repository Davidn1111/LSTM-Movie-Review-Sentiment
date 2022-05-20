"""
Main method for sentiment analysis of Yelp data
Base code provided by Huajie Shao, Ph.D
Code completed by Roger Clanton, David Ni, and Evan Ward
"""

import torch
from torch import nn
import pandas as pd 
import re
import matplotlib.pyplot as plt
import numpy as np
import json
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,TensorDataset
# import pandas as pd
from DataLoader import MovieDataset
from LSTM import LSTMModel
from GloveEmbed import _get_embedding
import time
from torch.utils.tensorboard import SummaryWriter


'''save checkpoint'''
def _save_checkpoint(ckp_path, model, epoches, global_step, optimizer):
    checkpoint = {'epoch': epoches,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, ckp_path)


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # torch.cuda.set_device(device=0) ## choose gpu number
    print('device: ', device)

    mode = 'train'
    Batch_size = 200
    n_layers = 1 ## choose 1-3 layers

    ## input seq length aligned with data pre-processing
    input_len = 150

    # word embedding length
    # equal to embedding dimensions of GloVe embeddings used)\
    embedding_dim = 200

    # lstm hidden dim
    hidden_dim = 100
    # binary cross entropy
    output_size = 1
    num_epoches = 1
    # learning rate
    learning_rate = 0.004
    # gradient clipping
    clip = 5
    load_cpt = False #True
    ckp_path = 'cpt/name.pt'
    # embedding_matrix = None
    ## use pre-train Glove embedding or not?
    pretrain = True

    # Define GloVe Embeddings file
    glove_file = 'glove.6B.200d.txt' # Change path accordingly

    ## step 1: create data loader in DataLoader.py
    
    ## step 2: load training and test data from data loader [it is Done]
    training_set = MovieDataset('training_data.csv')
    training_generator = DataLoader(training_set, batch_size=Batch_size,\
                                    shuffle=True,num_workers=1)

    test_set = MovieDataset('test_data.csv')
    test_generator = DataLoader(test_set, batch_size=Batch_size,\
                                shuffle=False,num_workers=1)

    ## step 3: Read tokens and load pre-train embeddings
    with open('tokens2index.json', 'r') as f:
        tokens2index = json.load(f)
    vocab_size = len(tokens2index)

    if pretrain:
        print('***** load glove embedding now...****')
        embedding_matrix = _get_embedding(glove_file,tokens2index,embedding_dim)
    else:
        embedding_matrix = None

    ## step 4: import model from LSTM.py
    model = LSTMModel(vocab_size=vocab_size,output_size=output_size,embedding_dim=embedding_dim,
                      embedding_matrix=embedding_matrix,hidden_dim=hidden_dim,n_layers=n_layers,
                      input_len=input_len,pretrain=pretrain)
    model.to()

    ## step 5: Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)  # 
    # Binary Cross Entropy Loss for binary classification problem
    loss_fun = nn.BCELoss()
    
    ## step 6: load checkpoint
    if load_cpt:
        # Terminal message
        print("*"*10+'loading checkpoint'+'*'*10)

        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoches = checkpoint["epoch"]

    ## step 7: model training
    print('*'*89)
    print('start model training now')
    print('*'*89)
    iteration = 0
    if mode == 'train':
        writer = SummaryWriter()

        model.train()
        for epoches in range(num_epoches):
            for x_batch, y_labels in training_generator:
                iteration += 1
                print(iteration)
                x_batch, y_labels = x_batch.to(device), y_labels.to(device)

                # Get model prediction
                y_out = model(x_batch)

                # Compute loss function
                loss = loss_fun(y_out,y_labels)
                writer.add_scalar("Loss/train", loss, iteration)

                ## step 8: back propagation
                optimizer.zero_grad()
                loss.backward()

                # Compute training accuracy
                y_pred = torch.round(y_out)
                acc = (y_pred == y_labels).sum().item() / len(y_pred)
                writer.add_scalar("Accuracy/train", acc, iteration)
                ## clip_grad_norm helps prevent the exploding gradient problem in LSTMs.
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                
        writer.flush()
        writer.close()

    ## step 9: Model testing
    print("----model testing now----")
    testAcc = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    model.eval()
    with torch.no_grad():
        for batch_id, (x_batch, y_labels) in enumerate(test_generator):
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)

            # Get model prediction
            y_out = model(x_batch)
            y_pred = torch.round(y_out)
            # Construct confusion matrix
            # tp += ((y_pred == y_labels) and (y_pred == 1)).sum().item()
            # tn += ((y_pred == y_labels) and (y_pred == 0)).sum().item()
            #
            # fp += ((y_pred != y_labels) and (y_pred == 1)).sum().item()
            # fn += ((y_pred != y_labels) and (y_pred == 0)).sum().item()

            #https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
            confusion_vector = y_pred / y_labels

            tp += torch.sum(confusion_vector == 1).item()
            fp += torch.sum(confusion_vector == float('inf')).item()
            tn += torch.sum(torch.isnan(confusion_vector)).item()
            fn += torch.sum(confusion_vector == 0).item()

            acc = (y_pred == y_labels).sum().item()/Batch_size
            print(acc)
            testAcc.append(acc)

        # Calculate accuracy, precision, recall, and f1 based on confusion matrix
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        f1 = 2*precision*recall / (precision+recall)
        print(f"True Positive: {tp}")
        print(f"True Negative: {tn}")
        print(f"False Positive: {fp}")
        print(f"False Negative: {fn}")

        print(f"Calculated Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print("running time: ", (time_end - time_start)/60.0, "mins")
    


    

    