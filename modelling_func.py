import pandas as pd
import numpy as np

from sklearn.metrics import SCORERS, precision_score, recall_score, \
    accuracy_score, f1_score, roc_curve, auc, confusion_matrix, roc_auc_score



def calculate_scores(y_train_true, y_train_pred, y_valid_true, y_valid_pred, m, verbose = False):
    """
    calculates scores, updates dictionary with relevant scores
    if verbose is true, also prints out results
    returns a dictionary
    """
    
    scoring_dictionary = {}
    
    scoring_dictionary['train_accuracy'] = accuracy_score(y_train_true, y_train_pred)
    scoring_dictionary['validation_accuracy'] = accuracy_score(y_valid_true, y_valid_pred)
    scoring_dictionary['train_f1'] = f1_score(y_train_true, y_train_pred)
    scoring_dictionary['validation_f1'] = f1_score(y_valid_true, y_valid_pred)
    scoring_dictionary['train_auc'] = roc_auc_score(y_train_true, y_train_pred)
    scoring_dictionary['validation_auc'] = roc_auc_score(y_valid_true, y_valid_pred)
    scoring_dictionary['train_zweigcampbell'] = \
        calculate_zweig_campbell_score(y_train_true, y_train_pred, m)
    scoring_dictionary['validation_zweigcampbell'] = \
        calculate_zweig_campbell_score(y_valid_true, y_valid_pred, m)
    
    if verbose:
        print("Train accuracy : " + str(scoring_dictionary['train_accuracy']))
        print("Validation accuracy : " + str(scoring_dictionary['validation_accuracy']))
        print("Train F1 : " + str(scoring_dictionary['train_f1']))
        print("Validation F1 : " + str(scoring_dictionary['validation_f1']))
        print("Train AUC : " + str(scoring_dictionary['train_auc']))
        print("Validation AUC : " + str(scoring_dictionary['validation_auc']))
        print("Train Zweig-Campbell : " + str(scoring_dictionary['train_zweigcampbell']))
        print("Validation Zweig-Campbell : " + str(scoring_dictionary['validation_zweigcampbell']))
        
    return scoring_dictionary
    
    
def calculate_zweig_campbell_score(y_true, y_pred, m):
    """
    calculates tpr - m * fpr
    """
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)

    return tpr - fpr * m