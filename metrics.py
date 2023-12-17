import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def Accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]


def Recall(y_true, y_pred):
      temp = 0
      for i in range(y_true.shape[0]):
          if sum(y_true[i]) == 0:
              continue
          temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_true[i])
      return temp/ y_true.shape[0]
  
def Precision(y_true, y_pred):
      temp = 0
      for i in range(y_true.shape[0]):
          if sum(y_pred[i]) == 0:
              continue
          temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_pred[i])
      return temp/ y_true.shape[0]
  
  
def F1Measure(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        temp+= (2*sum(np.logical_and(y_true[i], y_pred[i])))/ (sum(y_true[i])+sum(y_pred[i]))
    return temp/ y_true.shape[0]



def calculate_metrics(y_true, y_pred):
    acc = Accuracy(y_true, y_pred)
    prec = Precision(y_true, y_pred)
    rec = Recall(y_true, y_pred)
    f1 = F1Measure(y_true, y_pred)
    return acc, prec, rec, f1


def calculate_metrics_old(gt, preds):
    acc = accuracy_score(gt, preds)
    prec = precision_score(gt, preds, average = 'weighted')
    recall = recall_score(gt, preds, average = 'weighted')
    f1  = f1_score(gt, preds, average = 'weighted')
    return acc, prec, recall, f1