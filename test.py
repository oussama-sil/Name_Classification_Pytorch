import torch 





# Functions to test the model 

def test(model,test_loader,loss_fct,device,eval_fct):
    """
        Test the model on test data
    """

    with torch.no_grad():

        for i, (x,y,seq,label,lengths) in enumerate(test_loader):

            # Pushing the data to the device
            x,y = x.to(device),y.to(device)
                    
                    
            # Forward through the model
            y_hat = model(x,lengths)

            # Compute the loss
            # loss = loss_fct(y_hat,y)
            loss = loss_fct(y_hat, torch.argmax(y, dim=1)) #! NLLOSS accepts index of class 

            # Evaluate the model
            eval_ = eval_fct(y_hat,y)