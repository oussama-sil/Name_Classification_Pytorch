import torch
import torch.nn as nn
import os 
import time
import sys 

from data import construct_data
from model import RNN
from eval import accuracy

from torchinfo import summary

# Tensorboard 
from torch.utils.tensorboard import SummaryWriter 

#! Delete tensorflow warnings :
import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

torch.cuda.empty_cache()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"


def train(model,data,loss_fct,optimizer,nb_epochs,device,tracking_params,eval_fct,checkpoint_options,tensorboard_options):
    """
        Train the model for nb_epochs in device
        Input 
            model 
            data (tuple) : dataloader for train,val 
            loss_funct 
            optimizer 
            nb_epochs
            device 
            tracking_params : dict that contains the parameters for tracking the loss and display 
                -> track_loss_train : Bool # track the loss or not on the test data
                -> track_loss_train_steps : if track_loss_train, save the loss each .. steps (batch)
                -> track_loss_val : Bool # track the loss or not on the test data
                -> debug : print to the console the evolution of the training
                -> debug_loss : print the evolution of the loss in each epoch 
                -> average : record the average loss or the last loss
            eval_fct : a function to evaluate the model (acc, recall...), takes as inpu (y_hat,y) and returns an evaluation of the model
            checkpoint_options: dict that contains the parameters for checkpoint management 
                -> checkpoint : Bool # if True save a check point each 
                -> checkpoint_folder_path : Path to save the checkpoint
                -> checkpoint_epoch : # checkpoint each checkpoint_epoch epochs
            tensorboard_options : dict that contains the parameters for tensorboard 
                -> tensorboard : Bool # if True use tensorboard to print the evolution of the loss and evaluation 
                -> writer : str # path to the log directory
                -> label : label to add in the tensorboard 
        Output 
            loss_train : Evolution of the loss of the train
            epoch_loss : Loss for each epoch
            loss_val : Evolution of loss on validaton data 
            eval_loss : Evolution of eval metric on validatio data 
    """


    train_loader,val_loader = data
    
    
    nb_batch = len(train_loader) # NB batchs

    # For loss on train data each step
    loss_train = []
    tmp_loss_train = 0
    tmp_loss_train_count = 0
    
    # For loss on train data each epoch => to print the average loss during the epoch 
    epoch_loss = 0
    epoch_loss_count = 0


    # For loss on validation data
    loss_val = []
    eval_loss = []

    
    # For tensorboard 
    writer = tensorboard_options["writer"] or  None   
    if tensorboard_options["tensorboard"] : 
        pass


        
    print("\n Training the model")
    print(f"\t > Device : {device}")
    print(f"\t > NB_epochs : {nb_epochs}")
    print(f"\t > Batch size : {train_loader.batch_size}")
    print(f"\t > NB Batches : {nb_batch}")
    print(f"""\t > Debug : {'On' if tracking_params["debug"] else 'Off' }""")

    print()


    # Training loop 
    for epoch in range(nb_epochs):
        print(f"Epoch :  [{str(epoch+1):{3}} / {nb_epochs}]")
        start_time = time.time()

        for i, (x,y,seq,label,lengths) in enumerate(train_loader):
            # torch.cuda.empty_cache() # to train the large model 
            
            # Pushing the data to the device
            x,y = x.to(device),y.to(device)

            # Forward through the model
            y_hat = model(x,lengths)

            # print(y_hat.size())
            # print(y.size())


            # Compute the loss
            # loss = loss_fct(y_hat,y)
            loss = loss_fct(y_hat, torch.argmax(y, dim=1)) #! NLLOSS accepts index of class 


            # Backward 
            optimizer.zero_grad() # Set thegrads to zero
            loss.backward() # Compute the gradiants for all the parameters 
            optimizer.step() # Update the parameters 
            
            # Loss for the epoch
            epoch_loss += loss.item()
            epoch_loss_count += 1
            

            # Recoding the average loss on the train data and displaying it to the console 
            if tracking_params["track_loss_train"] or tracking_params["debug_loss"] :
                tmp_loss_train += epoch_loss
                tmp_loss_train_count += 1
                
                if i % tracking_params["track_loss_train_steps"] == 0 or i == nb_batch-1:
                    
                    if tracking_params["debug"] and tracking_params["debug_loss"]:
                        if tracking_params["average"]:  # Recording the average loss
                            print(f"\t -> Step:{str(i):{6}} ,  Loss = {tmp_loss_train / tmp_loss_train_count:.6f} ")
                            if tracking_params["track_loss_train"]:
                                loss_train.append(tmp_loss_train / tmp_loss_train_count)
                                if tensorboard_options["tensorboard"] : 
                                    writer.add_scalar(f"""Training Loss -{tensorboard_options["label"]}-""",tmp_loss_train / tmp_loss_train_count,epoch*nb_batch+i)
                        else: # Recording the last loss 
                            print(f"\t -> Step:{str(i):{6}} ,  Loss = {epoch_loss:.6f} ")
                            if tracking_params["track_loss_train"]:
                                loss_train.append(epoch_loss)
                                if tensorboard_options["tensorboard"] : 
                                    writer.add_scalar(f"""Training Loss -{tensorboard_options["label"]}- (steps)""",tmp_loss_train / tmp_loss_train_count,epoch*nb_batch+i)
                    
                    tmp_loss_train = 0
                    tmp_loss_train_count = 0

        end_time = time.time()
        training_time = end_time - start_time
        print("Evaluating")
        # Recording the loss and evaluation on the validation data at the end of each epoch 
        if tracking_params["track_loss_val"]:
            
            # Computing the loss and evaluating the model 
            with torch.no_grad():
                # Evaluate on one batch of test data
                for i, (x,y,seq,label,lengths) in enumerate(val_loader):
                    # Pushing the data to the device
                    x,y = x.to(device),y.to(device)

                    # Forward through the model
                    y_hat = model(x,lengths)

                    # Compute the loss
                    # loss = loss_fct(y_hat,y)
                    loss = loss_fct(y_hat, torch.argmax(y, dim=1)) #! NLLOSS accepts index of class 

                    # Evaluate the model
                    eval_ = eval_fct(y_hat,y)


                    loss_val.append(loss.item())
                    eval_loss.append(eval_)

                    if tensorboard_options["tensorboard"] : 
                        writer.add_scalar(f"""Val Loss -{tensorboard_options["label"]}- """,loss.item(),epoch)
                        writer.add_scalar(f"""Val eval  -{tensorboard_options["label"]}- """,eval_,epoch)

                    break


            if tracking_params["debug"] :
                print(f" -> End of epoch,  Train Loss = {epoch_loss / epoch_loss_count:.6f}, Val Loss = {loss.item():.6f}, Eval ={eval_} , t= {training_time:.2f} seconds \n")
                
        elif tracking_params["debug"] :
                print(f" -> End of epoch,   Train Loss = {epoch_loss / epoch_loss_count:.6f} , t= {training_time:.2f} seconds\n")
        
        
        if tensorboard_options["tensorboard"] : 
            writer.add_scalar(f"""Training Loss -{tensorboard_options["label"]}- (epochs)""",epoch_loss / epoch_loss_count,epoch)
        epoch_loss = 0
        epoch_loss_count = 0
        
        
        if checkpoint_options["checkpoint"] and (epoch+1) % checkpoint_options["checkpoint_epoch"] == 0: 
            # Saving the model, optimizer and epoch 
            checkpoint ={
                "epoch" : epoch, #current epoch
                "model_state" : model.state_dict(),
                "optimizer_state" : optimizer.state_dict(),
            }
            torch.save(checkpoint,os.path.join(checkpoint_options["checkpoint_folder_path"],f"checkpoint_{epoch}.pth"))

        if checkpoint_options["checkpoint"] and epoch == nb_epochs-1: # Last checkpoint 
            # Saving the model, optimizer and epoch 
            checkpoint ={
                "epoch" : epoch, #current epoch
                "model_state" : model.state_dict(),
                "optimizer_state" : optimizer.state_dict()
            }
            torch.save(checkpoint,os.path.join(checkpoint_options["checkpoint_folder_path"],f"checkpoint_{epoch}_final.pth"))

    
    return loss_train,epoch_loss,loss_val,eval_loss




#TODO : Tensorboard printing 
#TODO : Add load from checkpoint



if __name__ == "__main__":
    print("=======> Train.py <======")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 0.001
    batch_size = 32
    # Loading the data 
    vocab,country_index,index_country,list_country,train_data,test_data,val_data,train_loader,test_loader,val_loader = construct_data(dataset_folder="./data",batch_size=batch_size)


    # Model 
    model = RNN(input_size=len(vocab), hidden_size=512,num_layers=4,num_classes=len(list_country))
    # model.half()
    model.to(device)


    # Print model 
    writer = SummaryWriter("runs/RNN")
    x,y,seq,label,lengths = next(iter(train_loader))
    # x.bfloat16()
    writer.add_graph(model,x.to(device))
    

    # i = 0
    # for name, param in model.named_parameters():
    #     if i<52:
    #         i += 1
    #         param.requires_grad = False
    #         print(f"{name}: {param.requires_grad}")

    summary(model,input_data =x.to(device),device=device)   

    # Data for train function
    data = (train_loader,val_loader)

    # Loss function 
    loss_funct = nn.NLLLoss() #! Applies the softmax on the output 

    # Optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # NB epochs 
    nb_epochs = 8
    tracking_params ={
        "track_loss_train" : True, # Save loss on train data
        "track_loss_train_steps" : 50, # save loss each. .. step
        "track_loss_val" : True,
        "debug" : True,
        "debug_loss" : True,
        "average":True
    }

    eval_fct = accuracy

    #Check point options 
    checkpoint_options = {
        "checkpoint" : True,
        "checkpoint_folder_path" : ".\checkpoints",
        "checkpoint_epoch":10000
    }

    # Tensorboard options 
    tensorboard_options = {"tensorboard" : True, 
                "writer" : writer,
                "label" : "RNN" }

    train(model,data,loss_funct,optimizer,nb_epochs,device,tracking_params,accuracy,checkpoint_options,tensorboard_options)


    writer.close() 
    # sys.exit()
