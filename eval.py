import torch 



def accuracy(y_hat,y):
    """
        Evaluate the accuracy of the model
        Input : 
            y_hat [m,nb_classes]: probability distribution for each elem in the batch 
            y [m,nb_classes] : one hot encoding for the label for each element in the batch
    """
    _, predicted_classes = y_hat.max(dim=1)
    _, true_classes = y.max(dim=1)
    correct_predictions = (predicted_classes == true_classes).float().sum()
    accuracy = correct_predictions / y.size(0)
    return accuracy.item()
