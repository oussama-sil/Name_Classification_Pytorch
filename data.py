import os
from unidecode import unidecode
import random
import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#? To print vectore on one line on the console 
terminal_width = os.getenv('COLUMNS',200)
torch.set_printoptions(linewidth=terminal_width)



#? Dataset Class

#* One hot encoding for each letter 
def one_hot_embedding_word(word,vocab):
    """
        Construct the one hot encoding of a word
    """
    one_hot_tensor = torch.zeros(len(word), len(vocab), dtype=torch.float)

    for i, char in enumerate(word):
        index = vocab.index(char)
        one_hot_tensor[i, index] = 1

    return one_hot_tensor

def one_hot_embedding_label(label,list_country):
    """
        Construct the one hot encoding of a label
    """
    one_hot_tensor = torch.zeros(len(list_country), dtype=torch.int)
    one_hot_tensor[list_country.index(label)] = 1 
    return one_hot_tensor

#* Function that returns the function that transforms one element of the dataset
def transform(vocab,list_country):
    """
        Function that returns the function that transforms one element of the dataset
    """
    def transfrom_one_elem(elem ):
        """
            Transform one element of the dataset 
                x --> One hot encoding
                y --> Numerical encoding 
        """
        x,y = elem
        return one_hot_embedding_word(x,vocab),one_hot_embedding_label(y,list_country)
    return transfrom_one_elem

# load  file, 
class NamesDataset(Dataset):
    
    def __init__(self,path_data,max_seq_length=None,transform=None):

        self.transform = transform
        self.max_seq_length = max_seq_length

        # Open the file in read mode
        with open(path_data, 'r') as file:
            # Read all lines and store them in a list
            lines = file.readlines()
        
        self.x= [] # sequence
        self.y = [] # labels

        for line in lines:
            parts = line.strip().split()
            self.x.append(parts[0].lower())
            self.y.append(parts[1])

        # size of the data 
        self.m = len(self.x)

    def __getitem__(self, index):
        """
            Load one item, if large dataset use loading with openning file..
            How to load one element from the dataset 
        """
        if self.transform:
            if self.max_seq_length and len(self.x[index])>self.max_seq_length:
                self.x[index] = self.x[index][:self.max_seq_length]
            x_,y_ = self.transform((self.x[index],self.y[index]))
            return self.x[index],self.y[index],x_,y_
        else:
            if self.max_seq_length and len(self.x[index])>self.max_seq_length:
                self.x[index] = self.x[index][:self.max_seq_length]
            return self.x[index],self.y[index],None,None
    
    def __len__(self):
        return self.m

    def get_data(self):
        return self.x, self.y

    def get_vocab(self):
        """
            Return the vocab as a list of the letters 
        """
        vocab = set()

        for string in self.x:
            # Convert the string to lowercase and add its letters to the set
            vocab.update(char.lower() for char in string )

        # Convert the set to a sorted list
        vocab = sorted(list(vocab))

        return vocab

    def get_labels(self):
        """
            Returns the list of possible labels 
            Output : 
                - country_index : dict that maps country to its corresponding index
                - index_country : dict that maps index to its corresponding country
                - list_country : list of countries 
        """
        country_index= {}
        index_country={}
        for index, country in enumerate(set(self.y)):
            country_index[country] = index
            index_country[index] = country

        list_country = set()
        list_country.update(string for string in self.y )
        list_country = sorted(list(list_country))
        
        return country_index,index_country,list(country_index)


def collate_fn(batch):
    """
        This function is useful for handeling sequences of varying length
        => Performs padding of sequences 
        returns :
            padded_sequences : batch of data padded size : (m,max_length_sequence,embedding_size)
            y : labels one hot encoded 
            sequences : list of sequences as strings
            labels : labels as strings 
            lengths : lengths of the sequences 
    
    """
    # Separate data and labels
    sequences, labels,x,y = zip(*batch)
    # print(sequences)
    lengths = [x_.size()[0] for x_ in x]
    # Use pad_sequence to pad only the data sequences
    padded_sequences = pad_sequence(x, batch_first=True)
    # print(padded_sequences)
    # Convert labels to a tensor
    # label_tensor = torch.tensor(labels)
    
    return padded_sequences, y, sequences, labels,lengths


def construct_data(dataset_folder="./data",batch_size=4):

    """
        Function that returns the train, validation, and test datasets (dataset + loaders )
    """

    #? Load the full dataset to get the vocab and the list of countries 
    dataset = NamesDataset(os.path.join(dataset_folder,'dataset.txt'))
    vocab = dataset.get_vocab()
    country_index,index_country,list_country = dataset.get_labels()
    
    #? Data transformer => One hot encoding for labels and sequences
    transformer = transform(vocab,list_country)

    #? Load the train,val and test data 
    train_data = NamesDataset(os.path.join(dataset_folder,'train.txt'),transform=transformer)
    test_data = NamesDataset(os.path.join(dataset_folder,'test.txt'),transform=transformer)
    val_data = NamesDataset(os.path.join(dataset_folder,'val.txt'),transform=transformer)

    #? DataLoaders 
    train_loader = DataLoader(dataset=train_data,batch_size=4,shuffle=True,collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_data,batch_size=4,shuffle=False,collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_data,batch_size=4,shuffle=False,collate_fn=collate_fn)

    return vocab,country_index,index_country,list_country,train_data,test_data,val_data,train_loader,test_loader,val_loader



#? Parsing data
# Convert data from in folder to a single file containing : name label
def prepare_data(path_in="./data/names",path_out="./data",train_size=0.7,val_size=0.15,test_size=0.15):
    """
        Function that merges the different files to a single file 
        containing the full dataset , each record in the file 
        contains the tuple (name, country)
    """
    names = []
    for file_name in os.listdir(path_in):
        file_path = os.path.join(path_in,file_name)

        # Checking that the path is a file 
        if os.path.isfile(file_path):
            with open(file_path,'r', encoding='utf-8') as file:
                print(f" - Openning the file : {file_name}")
                for line_number, line_content in enumerate(file, start=1):
                    line_content_ascii = unidecode(line_content) # For languages such as french and spanish that have accents 

                    names.append(f"{line_content_ascii.strip().replace(' ', '')} {os.path.splitext(file_name)[0]}")
    
    names = random.sample(names, len(names)) # shuffle the order of the data
    
    print(" ---> Writing dataset")
    # Writing the full dataset to output file 
    path_dataset= os.path.join(path_out, "dataset.txt")
    with open(path_dataset, 'w', encoding='ascii') as output_file:
        # Write the transformed lines to the new file
        output_file.writelines("\n".join(names))

    print(" ---> Writing train data ")
    # Writing the train data to a file 
    path_train= os.path.join(path_out, "train.txt")
    train_data = names[:int(train_size*len(names))]
    with open(path_train, 'w', encoding='ascii') as output_file:
        # Write the transformed lines to the new file
        output_file.writelines("\n".join(train_data))

    print(" ---> Writing validation data ")
    # Writing the val data to a file 
    path_val= os.path.join(path_out, "val.txt")
    val_data = names[int(train_size*len(names)):int((train_size+val_size)*len(names))]
    with open(path_val, 'w', encoding='ascii') as output_file:
        # Write the transformed lines to the new file
        output_file.writelines("\n".join(val_data))

    print(" ---> Writing test data ")
    # Writing the test data to a file 
    path_test= os.path.join(path_out, "test.txt")
    test_data = names[int((train_size+val_size)*len(names)):int((train_size+val_size+train_size)*len(names))]
    with open(path_test, 'w', encoding='ascii') as output_file:
        # Write the transformed lines to the new file
        output_file.writelines("\n".join(test_data))







if __name__=="__main__":
    print("=======> Data.py <======")

    # prepare_data() 

    vocab,country_index,index_country,list_country,train_data,test_data,val_data,train_loader,test_loader,val_loader = construct_data(dataset_folder="./data",batch_size=6)

    examples = iter(train_loader)
    # print(next(examples)) # Get one batch from training data 
    # print( f"Shape of a batch X = {samples} Y = {labels} " ) 

    # dataset = NamesDataset('./data/dataset.txt')

    # vocab = dataset.get_vocab()
    # country_index,index_country,list_country = dataset.get_labels()

    # # print(len(dataset.get_vocab()))
    # # print(dataset.get_labels())

    # # dataloader = DataLoader(dataset=dataset,batch_size=4,shuffle=True,num_workers=1)#* num_workers : loading faster with multiple sub procesors
    
    # examples = iter(dataset)
    # print(next(examples)) # Get one batch from training data 
    # # print( f"Shape of a batch X = {samples} Y = {labels} " ) 

    # transformer = transform(vocab,list_country)

    # train_data = NamesDataset('./data/train.txt',transform=transformer)


    # dataloader = DataLoader(dataset=train_data,batch_size=4,shuffle=True,collate_fn=collate_fn)
    # examples = iter(dataloader)
    # x,y,seq,label,lengths = next(examples)


    # print(seq)
    # print(lengths)

    #? To work with RNN
    #"packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    #"print(packed_sequence.data.size())
    # print(x[0]) # Get one batch from training data
    # print(x[1]) # Get one batch from training data
    # print(x[2]) # Get one batch from training data
    # print(x[3]) # Get one batch from training data

    # print(list_country.index(labels))
    # print(transformer((samples, labels)))

    # print( one_hot_embedding(samples,vocab))
    # print( country_index[labels])
    # print( index_country[country_index[labels]])
    # print(list_country)

