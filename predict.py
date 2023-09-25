import onnxruntime
import torch
import json
from models.vae import VariationalAutoencoder




def predict_from_bin(path,
                    conf,
                    input=torch.tensor([
                        [[193,  29, 1]]
                        ]).float()):
    
    model = VariationalAutoencoder(conf)
    model.load_state_dict(torch.load(path))
    model.eval()
    output = model(input)
    return output


    