import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score

def calculate_metrics(gt, preds):
    acc = accuracy_score(gt, preds)
    prec = precision_score(gt, preds, average = 'weighted')
    recall = recall_score(gt, preds, average = 'weighted')
    f1  = f1_score(gt, preds, average = 'weighted')
    return acc, prec, recall, f1



# Train Loop-------------------------------------------------------------------------------------
def train_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0
    all_gts = []
    all_preds = []
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        
        with torch.enable_grad():
            outputs = model(data)
            #print("outputs: ", outputs)
            _, pred = torch.max(outputs, 1)
            #print("target: ", target)
            #print("target shape: ", target.shape)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
        
        
        running_loss += loss.item() * data.size(0)
        all_gts.extend(target.tolist())
        all_preds.extend(pred.tolist())
        
    acc, prec, recall, f1 = calculate_metrics(all_gts, all_preds)    
    return running_loss/len(train_loader), acc, prec, recall, f1

#Val Loop ----------------------------------------------------------------------------------------------------
def test_epoch(model, val_loader, loss_fn, device):
    model.eval()
    running_loss = 0
    all_gts = []
    all_preds = []
    for data, target in val_loader:
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            outputs = model(data)
            _, pred = torch.max(outputs, 1)
            loss = loss_fn(outputs, target)
            
        running_loss += loss.item() * data.size(0)
        all_gts.extend(target.tolist())
        all_preds.extend(pred.tolist())
        
    acc, prec, recall, f1 = calculate_metrics(all_gts, all_preds)    
    return running_loss/len(val_loader), acc, prec, recall, f1