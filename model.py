import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data import construct_data

# https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec : linkk  for padding

## RNN Models 
class RNN(nn.Module):
    def __init__(self,input_size, hidden_size,num_layers,num_classes) :
        super(RNN,self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.device = device

        #*  size of input : x-> batch_size, max_seq_length, input_size
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        
        #* Linear layer followed by activate
        self.fc = nn.Linear(hidden_size,num_classes)

    def forward(self,x,lengths):
        """
            Input : x of size [m,seq_length,input_size]
            Output : y_hat of size [m,num_classes]
            Don't apply the softmax
        """
        #* Init hidden state [nb_layers, size_batch, hidden_size]
        # h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(self.device)

        #* packing the sequences
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        #* Forward through the RNN , return  h : final hidden state for each elements in each layers, output contains the output for each timestep in last layer 
        packed_output, h = self.rnn(packed_sequence) # the output is packed
    
        #! print(h.size()) [nb_layers,size_batch,size_hidden_state]
    
        #* Get output without packing 
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        print(output.size())

        #* out contains the hidden state of all the elements in the seq
        #* out of size [batch_size, seq_length, hidden_size]
        out = h[-1] #? Last hidden state for each element 
        #*out (m,128)
        out = self.fc(out)

        return out
        #! Don't apply softmax caus the cross entropy loss applies the softmax loss



if __name__ == "__main__":
    print("=======> Model.py <======")
    vocab,country_index,index_country,list_country,train_data,test_data,val_data,train_loader,test_loader,val_loader = construct_data(dataset_folder="./data",batch_size=6)

    examples = iter(train_loader)
    # print(next(examples)[0].size()) # Get one batch from training data 

    x,y,seq,label,lengths = next(examples)

    model = RNN(input_size=len(vocab), hidden_size=128,num_layers=3,num_classes=len(list_country))
    
    out = model(x,lengths)
    
    # summary(model,input_size=next(examples)[0].size(),device="cpu")
