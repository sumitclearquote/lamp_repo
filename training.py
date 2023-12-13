from networks import ClassifierNet,ClassifierNet2, ResNet18, ResNet50, EfficientNetB4, \
                        train_epoch, test_epoch

from datasets import get_dataloader

from inference import get_results
import sys
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import albumentations as A
from torch import nn
import numpy as np
from PIL import Image
import pandas as pd
import wandb
import signal
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from warnings import filterwarnings as w
w('ignore')
import os
import argparse
from cfg import cfg
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
device = "cuda" if torch.cuda.is_available()  else "cpu"
print(f"{device} is available")

def log_print_metrics(loss, acc, prec, rec, f1, loader_type):
    if loader_type == "train":
        string  = f'Train Loss: {loss:.4f}, Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}'
    else:  
        string  = f'Val Loss: {loss:.4f}, Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}'
    
    print(string)
    with open(f"{log_dir}/log.txt", 'a') as f:
        f.write(string)
    
    
  

def train_model(data_dir, wandb, config, save_model = None):
    
    model_type=  config['model_type']
    model_name= config['model_name']
    lr= config["learning_rate"]
    batch_size = config['batch_size']
    company_name = config["company_name"]
    img_size= config["img_size"]
    weight_decay = config['weight_decay']
    save_model_dir = cfg.train.log_dir
    save_interval = cfg.train.save_interval
  

    data_mean  = cfg.data.data_mean
    data_std = cfg.data.data_std
    
    
    if config['Aug']:
        train_dl = get_dataloader(f"{data_dir}/train", batch_size, data_mean,
                                data_std, img_size, dataset_type = "train", weights_sampler = True)
    else: # Use same augs as val
        train_dl = get_dataloader(f"{data_dir}/train", batch_size, data_mean,
                                data_std, img_size, dataset_type = "val", weights_sampler = True)
    
    val_dl = get_dataloader(f"{data_dir}/train", batch_size, data_mean,
                              data_std, img_size, dataset_type = "val", weights_sampler = False)


    if model_type == "resnet18":
        print("Loading ResNet18 ..")
        model = ResNet18(num_classes, pretrained = cfg.train.pretrain)
    elif model_type == "resnet50":
        model = ResNet50(num_classes, pretrained= cfg.train.pretrain)
    elif model_type == "efficientnetb4":
        model = EfficientNetB4(num_classes, pretrained= cfg.train.pretrain)
    

    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    
    print("\nTraining ..")
    for epoch in tqdm(range(n_epochs)):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(model, train_dl, loss_fn, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = test_epoch(model, val_dl, loss_fn, device)
        scheduler.step(val_loss)
        
        
        if wandb:
            # Log metrics to wandb
            wandb.log({"Train/Loss": train_loss, "Train/Accuracy": train_acc, "Train/Precision":train_prec, "Train/Recall":train_rec, "Train/F1":train_f1}, step = epoch)
            wandb.log({"Val/Loss": val_loss, "Val/Accuracy": val_acc, "Val/Precision":val_prec, "Val/Recall":val_rec, "Val/F1":val_f1}, step = epoch)
            wandb.log({"Learning_Rate": optimizer.param_groups[0]['lr']}, step = epoch)
                                                                
        if (epoch+1) % 1 == 0:
            print("\nEpoch: ", epoch)
            log_print_metrics(train_loss, train_acc, train_prec, train_rec, train_f1, loader_type='train')
            log_print_metrics(val_loss, val_acc, val_prec, val_rec, val_f1, loader_type='val')
            print("----")
            
        if epoch!=0 and epoch % save_interval==0 and save_model:
            torch.save(model, f"{save_model_dir}/{model_name}_{epoch}.pth")
            torch.save(model.state_dict(), f"{save_model_dir}/{model_name}_{epoch}_sd.pth")
            
    if wandb:   
        wandb.finish()  
        
    if save_model:   
        torch.save(model, f"{save_model_dir}/{model_name}_latest.pth")
        torch.save(model.state_dict(), f"{save_model_dir}/{model_name}_sd_latest.pth")

    return model

def signal_handler(signal, frame):
    # Handle the interrupt signal
    if wandb:
        wandb.finish()
    print("Program interrupted. Exiting...")
    sys.exit(0)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = "Script to perform inference on unseen test set")
    
    parser.add_argument('--model_type', '-m',required= False, choices = ["resnet18", "resnet50", "customnet", "newcustomnet", "mobnetv3"], help = "Model Type")
    parser.add_argument('--uniqueid', '-u',required= False, help = "Unique identifier for the run (wandb)")
    parser.add_argument('--use_aug', '-a', action = 'store_true', help = "Color Name")
    parser.add_argument('--save_model', '-s', action = 'store_true', help = "Save Model?")
    parser.add_argument('--use_wandb', '-w', action = 'store_true', help = "Log results to wandb")
    parser.add_argument('--batch_size', '-b', required =True,  help = "Training batch size")
    parser.add_argument('--imsize', '-ims', required = True,help = "Img size to train on" )
    '''
    parser.add_argument('--company_name', '-c', required= True, help = "Paint manufacturer name")
    parser.add_argument('--color', '-col', default = 'magmagrey', help = "Color Name")
    
    '''
    #Command line arguments
    #cla = parser.parse_args()
    #model_type = cla.model_type   #args.model_type
    #unique_id =  cla.uniqueid
    #use_aug =   cla.use_aug
    #im_size =   int(cla.imsize)
    #save_model = cla.save_model
    #use_wandb =  cla.use_wandb 
    
    
    
    
    #Config Arguments
    args = cfg
    
    # Other cfg parameters
    unique_id =  args.train.uniqueid
    model_type = args.train.model_type
    use_wandb = args.train.use_wandb
    use_aug = args.train.use_aug
    save_model  = args.train.save_model
    
    data_dir = args.data.dataset_dir
    
    img_size = args.data.img_size
    
    classes = args.train.classes
    
    log_dir = args.train.log_dir

    #Hyper parameters
    n_epochs = args.train.epochs #60
    batch_size = int(args.train.batch_size) 
    lr = args.train.lr
    weight_decay = args.train.weight_decay
    

  
    torch.cuda.empty_cache()
    
    # Create config
    model_name  = f"{model_type}_{unique_id}"
    config = {"img_size":img_size,  "learning_rate" : lr, "weight_decay": weight_decay, "epochs" : n_epochs, "batch_size" : batch_size, "model_name":model_name, 
              "Aug":use_aug, "model_type": model_type, "Desc":f"Trained {model_type} on {unique_id}"}
    
    #Initialize wandb---------------
    if use_wandb:
        wandb.init(project = "Lamp-Damages", name = model_name)# , id = "", resume = "must") #----------- 
        wandb.config.update(cfg) #------------#
    else:
        wandb = []
        
    print("Model Name: ", model_name)
    
    num_classes = len(classes)
    
    #Stops wandb logging when Ctrl+C is pressed
    signal.signal(signal.SIGINT, signal_handler)
    
    #Create log dir and log the config
    text = f"{model_name} (epochs: {n_epochs}):-------------------\n"
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/log.txt", 'a') as f:
        f.write(text)
        f.write(cfg.pretty_text)

    
    model = train_model(data_dir, wandb, config, save_model =save_model)
    
            
    #test_model(model, config)
