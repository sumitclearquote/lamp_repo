import numpy as np
import torch
from metrics import calculate_metrics, calculate_classwise_metrics


def operation(subarray, threshold = 0.5):
    '''
    Function to apply on a subarray of an array. 
    '''
    return [1 if i > threshold else 0 for i in subarray]



# Train Loop-------------------------------------------------------------------------------------
def train_epoch(model, train_loader, loss_fn, optimizer, device):
    '''
    Trains a single epoch
    '''
    model.train()
    running_loss = 0
    all_gts = []
    all_preds = []
    class_gt_preds = {'cracked': {'gts': [], 'preds': []}, 'fadelamp':  {'gts': [], 'preds': []}, 'foggy':  {'gts': [], 'preds': []}}
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        with torch.enable_grad():
            outputs = model(data)

            preds = np.apply_along_axis(operation,  axis = 1, arr = outputs.detach().cpu().numpy(), threshold = 0.5)
            labels = target.cpu().numpy()
     
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * data.size(0)
        
        if i==0:
            all_gts = labels; all_preds = preds
        if i!=0:
            all_gts = np.concatenate((all_gts, labels), axis = 0)
            all_preds = np.concatenate((all_preds, preds), axis = 0)
      
            
    class_gt_preds['cracked']['gts'].append(all_gts[:, 0]);     class_gt_preds['fadelamp']['gts'].append(all_gts[:, 1]);     class_gt_preds['foggy']['gts'].append(all_gts[:, 2])
    class_gt_preds['cracked']['preds'].append(all_preds[:, 0]); class_gt_preds['fadelamp']['preds'].append(all_preds[:, 1]); class_gt_preds['foggy']['preds'].append(all_preds[:, 2])

    class_metrics = calculate_classwise_metrics(class_gt_preds)
    
    acc, prec, recall, f1 = calculate_metrics(all_gts, all_preds) 
    
    return running_loss/len(train_loader), acc, prec, recall, f1, class_metrics

#Val Loop ----------------------------------------------------------------------------------------------------
def test_epoch(model, val_loader, loss_fn, device):
    '''
    Tests a single epoch
    '''
    model.eval()
    running_loss = 0
    class_gt_preds = {'cracked': {'gts': [], 'preds': []}, 'fadelamp':  {'gts': [], 'preds': []}, 'foggy':  {'gts': [], 'preds': []}}
    for i, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            outputs = model(data)
            preds = np.apply_along_axis(operation,  axis = 1, arr = outputs.detach().cpu().numpy(), threshold = 0.5)
            labels = target.cpu().numpy()
            loss = loss_fn(outputs, target)
            
        running_loss += loss.item() * data.size(0)
        if i==0:
            all_gts = labels; all_preds = preds
        if i!=0:
            all_gts = np.concatenate((all_gts, labels), axis = 0)
            all_preds = np.concatenate((all_preds, preds), axis = 0)
            
            
            
    class_gt_preds['cracked']['gts'].append(all_gts[:, 0]);     class_gt_preds['fadelamp']['gts'].append(all_gts[:, 1]);     class_gt_preds['foggy']['gts'].append(all_gts[:, 2])
    class_gt_preds['cracked']['preds'].append(all_preds[:, 0]); class_gt_preds['fadelamp']['preds'].append(all_preds[:, 1]); class_gt_preds['foggy']['preds'].append(all_preds[:, 2])

    class_metrics = calculate_classwise_metrics(class_gt_preds)
        
    acc, prec, recall, f1 = calculate_metrics(all_gts, all_preds)    
    return running_loss/len(val_loader), acc, prec, recall, f1, class_metrics