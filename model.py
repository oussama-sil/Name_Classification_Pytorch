import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# from torchsummary import summary
from torchinfo import summary
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data import construct_data

from utils import ConsoleColors

# https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec : linkk  for padding

## RNN Models 
class RNN(nn.Module):
    def __init__(self,input_size, hidden_size,num_layers,num_classes) :
        super(RNN,self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        #*  size of input : x-> batch_size, max_seq_length, input_size
        self.rnn = nn.GRU(input_size,hidden_size,num_layers,batch_first=True)
        
        #* Linear layer followed by activate
        self.fc = nn.Linear(hidden_size,hidden_size)
        self.fc_ = nn.Linear(hidden_size,num_classes)

        #* LogSoftmax layer for negative loglikelihood 

        self.softmax = nn.LogSoftmax(dim = 1) 

    def forward(self,x,lengths=None):
        """
            Input : x of size [m,max_seq_length,input_size] #Padded 
            Output : y_hat of size [m,num_classes] 
            Outputs are not probabilities 
        """
        # print("--")
        # print(x.size())
        # print(lengths)
        # print("--")

        if lengths == None:
            print(ConsoleColors.WARNING + " Error : length not provided , assuming all sequences of same length" + ConsoleColors.ENDC)
            lengths = [x.size()[1]]*x.size()[0]

        #* packing the sequences
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        #* Forward through the RNN , 
        # * return  :
        # *     h : final hidden state for each elements in each layers, [nb_layers,size_batch,size_hidden_state]
        # *     output : contains the output for each timestep in last layer 
        packed_output, h = self.rnn(packed_sequence) # the output is packed
    
        
        # print(h.size()) # [nb_layers,size_batch,size_hidden_state]
    
        #* Unpacking the output ==> out of size [m,max_seq_length,hidden_size]
        output, _ = pad_packed_sequence(packed_output, batch_first=True) #! Output size of [batch_size, seq_length, hidden_state length]

        # print(output.size())
        # print(_.size())

        # print(output[0])
        # print(x.size())

        # print(out)
        #* out contains the hidden state of all the elements in the seq
        #* out of size [batch_size, seq_length, hidden_size]
        out = h[-1] #? Last hidden state for each element 
        #*out (m,128)
        out = self.fc(out)
        out = self.fc_(out)
        out = self.softmax(out)
        return out
        #! Don't apply softmax because the cross entropy loss applies the softmax automatically



if __name__ == "__main__":
    print("=======> Model.py <======")
    vocab,country_index,index_country,list_country,train_data,test_data,val_data,train_loader,test_loader,val_loader = construct_data(dataset_folder="./data",batch_size=6)

    
    #? Dataset 
    # examples = iter(train_data)
    # seq,label,x,y = next(examples)
    # print(x.size()) # Embedding of one sequence [seq_length,embd_size] 
    # print(y.size()) # Embedded of label [nb_classes] (one hot encoding ) 
    # print(seq) # Sequence
    # print(label) # Label


    #? Dataloader
    examples = iter(train_loader)
    x,y,seq,label,lengths = next(examples)
    # print(x.size()) # Embedded Padded sequence [m,max_seq_length,embd_size] 
    # print(y.size()) # Embedded packed labels [m,nb_classes] (one hot encoding ) 
    # print(seq) # Tuple containing the sequences 
    # print(label) # Tuple containing the sequence labels 
    # print(lengths) # Array of length of each sequence #? For RNN 



    # Model
    #! Add .to(device)
    model = RNN(input_size=len(vocab), hidden_size=128,num_layers=6,num_classes=len(list_country))
    # out = model(x,lengths)
    # print(out.size())
    # Model summary 
    # summary(model,input_size=(10,len(vocab)),device="cpu")
    summary(model,input_data =x,device="cpu")
