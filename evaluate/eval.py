import torch
from utils import reconstruction_error_metric
def evaluation(model,val_dataloader,device):
    model=model.to_device()
    model.eval()
    singularity_error=[]
    for batch in val_dataloader:
        batch=batch.to(device)
        with torch.no_grad():
            output=model(batch)
    sa=reconstruction_error_metric(batch,output)
    singularity_error.append(sa) 
    return sa    
