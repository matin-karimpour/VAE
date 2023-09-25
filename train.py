import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Dict
from models.vae import VariationalAutoencoder
import dataset.data as data
import json
import pandas as pd
import onnx
from predict import predict_from_bin


def init_model(conf:Dict, device: str):
    print("Initialing Two-Tower Model...")
    model = VariationalAutoencoder(conf).to(device)
    return model





def train(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    print(size)
    model.train()
    losses = []
    for batch, x in enumerate(dataloader):
        optimizer.zero_grad()
        # Compute prediction error
        pred, _ = model(x)
        
        loss = ((x - pred)**2).sum() + model.kl
        losses.append(loss.item())
        # Backpropagation
        loss.backward()
        optimizer.step()
        

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        



def test(dataloader, model,):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X in dataloader:
            pred, _ = model(X)
            test_loss += (((X - pred)**2).sum() + model.kl).item()
    test_loss /= num_batches
    print(f"Test Error: \n , Avg loss: {test_loss:>8f} \n")
    return test_loss


def save_onnx_model(model, sample_input, save_path="./best_VAE.onnx",):
    print("Exporting model to ONNX...")
    torch.onnx.export(model,               # model being run
                  sample_input,                         # model input (or a tuple for multiple inputs)
                  save_path,   # where to save the model (can be a file or file-like object)
                  input_names = ['input_user', 'input_item'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model saved in ", save_path)
    


def fit(model, 
        train_dataloader, 
        test_dataloader, 
        epochs: int=5, 
        model_save_path: str="best_model.bin",
        patience: int = 10,
        ):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    test_losses = []
    early_stop_counter = 0
    print("Training Variational Autoencoder Model...")

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, optimizer)

        test_loss = test(test_dataloader, model)
        test_losses.append(test_loss)
        early_stop_counter +=1
        
        if test_loss <= min(test_losses):
            best_model = model
            torch.save(best_model.state_dict(), model_save_path)
            print(f"saved {model_save_path}")
            early_stop_counter = 0

        if early_stop_counter >= patience:
            break
    print("Training Done!")
    return best_model
    

def train_model(conf,
                   epochs: int = 5,
                   df: pd.DataFrame = data.ml100k_dataset(),
                   save_dir: str = "",
                   model_name: str = "best_model.bin",
                   save_onnx: bool = True,
                   patience:int = 10):
    
    model_save_path = save_dir + model_name
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")

    model = init_model(conf,device)

    features = conf["features"]
    

    train_dataloader, test_dataloader = data.get_train_test(df, 
                   features, 
                   device )
    
    
    
    best_model = fit(model, 
                     train_dataloader, 
                     test_dataloader,
                     epochs=epochs,
                     model_save_path=model_save_path,
                     patience=patience)
    
    # if save_onnx:
    #     best_model.eval()
    #     (sample_input_user,sample_input_item), torch_out = next(iter(test_dataloader))
    #     sample_input = [torch.reshape(sample_input_user[0],(1,-1)),
    #                     torch.reshape(sample_input_item[0],(1,-1))]
        
    #     print(sample_input)
    #     onnx_model_path = save_dir + model_name.split(".")[0] + ".onnx"
    #     save_onnx_model(best_model, sample_input,save_path=onnx_model_path)

    #     ort_session = init_onnx_model(onnx_model_path)
    #     out = predict_from_onnx(ort_session, sample_input)
    #     print(out)




if __name__=="__main__":

    with open('config/vae.json', 'r') as f:
        conf = json.load(f)
    train_model(conf,epochs=20)
    print(predict_from_bin("best_model.bin", conf))