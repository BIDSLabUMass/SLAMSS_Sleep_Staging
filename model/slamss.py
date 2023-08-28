import os
import csv
import pandas as pd
import numpy as np
import random
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import scipy.io
from sklearn.metrics import confusion_matrix
import glob
from torchvision.datasets.folder import IMG_EXTENSIONS
import time
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
    
        self.cnn = nn.Sequential( 
        nn.Conv1d(in_w, 64, 9, stride=1, padding=4),
        nn.LeakyReLU(inplace=False),
        nn.Conv1d(64, 64, 9, stride=1, padding=4),
        nn.LeakyReLU(inplace=False),
        nn.Conv1d(64, 64, 9, stride=1, padding=4),
        #nn.LeakyReLU(inplace=False), #put it back 2020 706
        )
        
    def forward(self, src):
        
        #src = [src sent len, batch size]
        
        #embedded = self.dropout(self.embedding(src))
        
        #embedded = [src sent len, batch size, emb dim]

        src1 = src.permute(0, 2, 1)
        #print src1.size()
        src2 = self.cnn(src1)
        #print src2.size()
        src3 = src2.permute(0, 2, 1)
        #print src3.size()

    
    
        return src3
    
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear((enc_hid_dim * 1) + dec_hid_dim, dec_hid_dim, bias=True)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        #repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        #encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src sent len, dec hid dim]
        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]


        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        #print energy
        #energy = [batch size, src sent len, dec hid dim]
        
        energy = energy.permute(0, 2, 1)
        
        #energy = [batch size, dec hid dim, src sent len]
        
        #v = [dec hid dim]
        
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        
        #v = [batch size, 1, dec hid dim]
     
        attention = torch.bmm(v, energy).squeeze(1)
        
        #attention= [batch size, src len]

        
        return F.softmax(attention, dim=1)



class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        #ORINAL 1 # 5
        self.rnn = nn.GRU(64, enc_hid_dim, 3, bidirectional = False, dropout=0.5, batch_first = True)
        
        self.fc = nn.Sequential( 
            nn.Linear(enc_hid_dim * 1, dec_hid_dim, bias=True),
            #nn.BatchNorm1d(dec_hid_dim),
            nn.Dropout(0.5),
        )
     
    def forward(self, src):
        
        #src = [src sent len, batch size]
        

        outputs, hidden = self.rnn(src)
  
        #hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        hidden = self.fc(hidden[-1,:,:])
        #hidden = self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        #outputs = [src sent len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        #print outputs
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super(Decoder, self).__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        #self.dropout = dropout
        self.attention = attention
        
        #self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 1) + emb_dim, dec_hid_dim,batch_first = True, dropout=0.0)
        
        self.out = nn.Sequential( 

            nn.Linear((enc_hid_dim * 1) + dec_hid_dim + emb_dim, output_dim, bias=False),
            #nn.Dropout(0.5),
        )
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, input, hidden, encoder_outputs, trans):
             
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        
        a = self.attention(hidden, encoder_outputs)
        #print a        
        #a = [batch size, src len]
    
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        encoder_outputs = encoder_outputs.permute(0, 1, 2)
        
        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        

        rnn_input = torch.cat((input, weighted), dim = 2)
        rnn_input = rnn_input.permute(1, 0, 2)
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
        #print hidden.size()

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        #print output 
        #output = [sent len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #sent len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden

        #assert (output == hidden).all()
        
        #input = input.squeeze(0)
        output = output.squeeze(1)
        output = output.unsqueeze(0)
        #weighted = weighted.squeeze(0)
        #print(output.size())
        #print(weighted.size())
        #print(input.size())
        #print(trans.size())
        #trans = trans.unsqueeze(0)
        #output = self.out(torch.cat((output, weighted, input, trans), dim = 2))
        output = self.out(torch.cat((output, weighted, input), dim = 2))
        #output = [bsz, output dim]
        
        return output, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device = 0):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        tn = scipy.io.loadmat("./transition.mat")
        trans = tn['transition_train_raw']
        self.trans = Variable(torch.FloatTensor(trans).cuda())
        #print(trans)
        #assert encoder.hid_dim == decoder.hid_dim, \
        #    "Hidden dimensions of encoder and decoder must be equal!"
        #assert encoder.n_layers == decoder.n_layers, \
        #    "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[0]
        max_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, max_len-1, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs, hidden = self.encoder(src)
        #print hidden.size()
        #first input to the decoder is the <sos> tokens
        input = trg[:,0]
        _, ind = torch.max(input,1)
        trans_bat = torch.zeros(batch_size, 5).to(self.device)
        for i in range(batch_size):
            trans_bat[i,:] = self.trans[ind[i],:]
        #print(input.size())
        #print(input2.size())
        
        for t in range(1, max_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.decoder(input, hidden, encoder_outputs, trans_bat)

            #place predictions in a tensor holding predictions for each token
   
            if output.size()[0] == 1:
                outputs[:,t-1,:] = output[:,:]
            else:	
                outputs[:,t-1,:] = output[:,0,:]
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            #need to chekc again   tzuan
            top1 = output.argmax(1)
            top1 = top1.view(-1,1).type(torch.cuda.FloatTensor)
            #print "Tp: ", top1.size()
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[:,t] if teacher_force else output.squeeze(0)
        

        return outputs

class features_dataset(Data.Dataset):

    def __init__(self, mode, p):

        self.mode = mode
        self.data = glob.glob(p)

    def __getitem__(self, index):


        if self.mode == 0:
            data = scipy.io.loadmat(self.data[index])
            x = np.squeeze(data['sqe_x'])[:800,:]
            y = data['sqe_y'][:800,:]
        elif self.mode == 1:
            data = scipy.io.loadmat(self.data[index])
            x = np.squeeze(data['sqe_x'])
            y = data['sqe_y']
            up_lim = x.shape[0] - 700
            ind = random.randint(sqe, up_lim)
            tmp_x = x[(ind-sqe):ind+700,:]
            tmp_y = y[(ind-sqe):ind+700,:]
            x = tmp_x#.reshape((sqe+50, w)) + np.random.normal(0,0.000001, size=(50+sqe, w))
            y = tmp_y

        else:
            data = scipy.io.loadmat(self.data[index])
            x = np.squeeze(data['sqe_x'])
            y = data['sqe_y']
            tem_x = np.zeros((2953,3))
            tem_y = np.zeros((2953,1))
            tem_x[:x.shape[0],:] = x[:,[1,6,7]]
            tem_y[:y.shape[0],:] = y[:,:]
            x1 = torch.FloatTensor(tem_x) 
            y1 = torch.LongTensor(tem_y)
            y2 = torch.zeros(x1.size()[0],cls).scatter_(dim=1, index=y1, src=torch.tensor(1.0))
            
            return x1, y1, y2, x.shape[0]
        #x = self.input_x[index].reshape((h, w))
        #y = self.input_y[index]			
        #x = x[:,[0, 1, 6]]	
        x = x[:,[1,6,7]]
        x1 = torch.FloatTensor(x) 
        y1 = torch.LongTensor(y)#.unsqueeze(1)
        
        
        #print(y1.size(), "                    "  ,x1.size())

        #print(y1.size())
        y2 = torch.zeros(x1.size()[0],cls).scatter_(dim=1, index=y1, src=torch.tensor(1.0))
        #print y2.size()
        #quit()

        return x1, y1, y2
    def __len__(self):
        if self.mode == 1:
            return(len(self.data))
        else:
            return(len(self.data))
            #return(len(self.x))
    def getName(self):
        return self.classes
