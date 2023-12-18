from networks import ResNet18, ResNet50, EfficientNetB4

from train_utils import train_epoch, test_epoch

from datasets import get_dataloader

import sys
import pandas as pd
import shutil
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
#import albumentations as A
from torch import nn
import numpy as np
from PIL import Image, ImageFile
import wandb
import signal
from tqdm import tqdm
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from warnings import filterwarnings as w
w('ignore')
import os
import argparse
from cfg import cfg
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


ImageFile.LOAD_TRUNCATED_IMAGES = True
device = "cuda" if torch.cuda.is_available()  else "cpu"
print(f"{device} is available")

def log_print_metrics(epoch, loss, acc, prec, rec, f1, loader_type):
    if loader_type == "train":
        string  = f'\nEpoch: {epoch}\nTrain Loss: {loss:.4f}, Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}\n'
    else:  
        string  = f'Val Loss: {loss:.4f}, Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}\n'
    
    print(string)
    with open(f"{log_dir}/log.txt", 'a') as f:
        f.write(string)
    
    
  

def train_model(data_dir, traindf, valdf, wandb, config, save_model = None):
    
    model_type=  config['model_type']
    model_name= config['model_name']
    lr= config["learning_rate"]
    batch_size = config['batch_size']
    img_size= config["img_size"]
    weight_decay = config['weight_decay']
    save_model_dir = cfg.train.log_dir
    save_interval = cfg.train.save_interval
  

    data_mean  = cfg.data.data_mean
    data_std = cfg.data.data_std
    
    train_data = traindf.values
    val_data = valdf.values
    
    if config['Aug']:
        train_dl = get_dataloader(f"{data_dir}/train",train_data, batch_size, data_mean,
                                data_std, img_size, dataset_type = "train", weights_sampler = True)
    else: # Use same augs as val
        train_dl = get_dataloader(f"{data_dir}/train",train_data, batch_size, data_mean,
                                    data_std, img_size, dataset_type = "val", weights_sampler = True)
    
    val_dl = get_dataloader(f"{data_dir}/val",val_data, batch_size, data_mean,
                              data_std, img_size, dataset_type = "val", weights_sampler = False)


    if model_type == "resnet18":
        print("Loading ResNet18 ..")
        model = ResNet18(num_classes, pretrained = cfg.train.pretrain)
    elif model_type == "resnet50":
        print("Loading ResNet50 ..")
        model = ResNet50(num_classes, pretrained= cfg.train.pretrain)
    elif model_type == "efficientnetb4":
        print("Loading EfficientNetB4 ..")
        model = EfficientNetB4(num_classes, pretrained= cfg.train.pretrain)
    

    model.to(device)
    loss_fn = nn.BCELoss()  #nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = cfg.train.patience,
                                                           factor = cfg.train.lr_factor) 
    
    
    print("\nTraining ..")
    for epoch in tqdm(range(n_epochs)):
        train_loss, train_acc, train_prec, train_rec, train_f1, class_train_metrics = train_epoch(model, train_dl, loss_fn, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, class_val_metrics = test_epoch(model, val_dl, loss_fn, device)
        scheduler.step(val_loss)
        
        
        if wandb:
            # Log metrics to wandb
            for dat in ["Train", "Val"]:
                for class_name in classes: #["cracked", "fadelamp", "foggy"]
                    if dat == "Train":
                        wandb.log({f"{class_name.capitalize()}/{dat}/Accuracy" : class_train_metrics[class_name][0],f"{class_name.capitalize()}/{dat}/Precision" : class_train_metrics[class_name][1], f"{class_name.capitalize()}/{dat}/Recall":class_train_metrics[class_name][2],f"{class_name.capitalize()}/{dat}/F1":class_train_metrics[class_name][3],}, step = epoch)
                    elif dat == "Val":
                        wandb.log({f"{class_name.capitalize()}/{dat}/Accuracy" : class_val_metrics[class_name][0],f"{class_name.capitalize()}/{dat}/Precision" : class_val_metrics[class_name][1], f"{class_name.capitalize()}/{dat}/Recall":class_val_metrics[class_name][2],f"{class_name.capitalize()}/{dat}/F1":class_val_metrics[class_name][3],}, step = epoch)
                        
            
            wandb.log({"Train/Loss": train_loss, "Train/Accuracy": train_acc, "Train/Precision":train_prec, "Train/Recall":train_rec, "Train/F1":train_f1}, step = epoch)
            wandb.log({"Val/Loss": val_loss, "Val/Accuracy": val_acc, "Val/Precision":val_prec, "Val/Recall":val_rec, "Val/F1":val_f1}, step = epoch)
            wandb.log({"Learning_Rate": optimizer.param_groups[0]['lr']}, step = epoch)
                                                                
        if (epoch+1) % 1 == 0:
            log_print_metrics(epoch, train_loss, train_acc, train_prec, train_rec, train_f1, loader_type='train')
            log_print_metrics(epoch, val_loss, val_acc, val_prec, val_rec, val_f1, loader_type='val')
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
    
    #Config Arguments
    args = cfg
    
    # Other cfg parameters
    unique_id =  args.train.uniqueid
    model_type = args.train.model_type
    use_wandb = args.train.use_wandb
    use_aug = args.train.use_aug
    save_model  = args.train.save_model
    
    data_dir = args.data.dataset_dir
    
    train_csv_path = args.data.train_path
    val_csv_path = args.data.val_path
    
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
    
    # Read traindf and valdf
    traindf = pd.read_csv(train_csv_path)
    valdf = pd.read_csv(val_csv_path)
    
    num_classes = len(classes)
    
    #Stops wandb logging when Ctrl+C is pressed
    signal.signal(signal.SIGINT, signal_handler)
    
    #create log dir
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy('cfg.py', f"{log_dir}/cfg.py")

    #log the config
    text = f"{model_name} (epochs: {n_epochs}):-------------------\n"
    with open(f"{log_dir}/log.txt", 'a') as f:
        f.write(text)

    
    model = train_model(data_dir, traindf, valdf, wandb, config, save_model =save_model)
    
