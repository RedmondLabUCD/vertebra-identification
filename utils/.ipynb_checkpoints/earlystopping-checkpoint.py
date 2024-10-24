import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Adapted from: github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    
    """
    
    def __init__(self, patience=5, verbose=False, up=True, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation accuracy improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation accuracy improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        self.trace_func = trace_func
        self.up = up
        if self.up:
            self.val_acc_worst = 0
            self.delta = 0.00001
        else:
            self.val_acc_worst = 1
            self.delta = 0.1
    
    def __call__(self, val_acc, model, optimizer, model_name, extra, epoch):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, optimizer, model_name, extra, epoch)
        elif (score < self.best_score and self.up) or (score > self.best_score and not self.up):
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model, optimizer, model_name, extra, epoch)
            self.counter = 0

    def save_checkpoint(self, val_acc, model, optimizer, model_name, extra, epoch):
        '''Saves model when validation accuracy increases.'''
        if self.verbose:
            self.trace_func(f'Validation accuracy increased ({self.val_acc_worst:.6f} --> {val_acc:.6f}).  Saving model ...')
        checkpoint_dict = {
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint_dict, os.path.join(self.path,"chkpt_{}".format(model_name+extra)))
        self.val_acc_worst = val_acc
        
    def load_checkpoint(optimizer, model, filename):
        checkpoint_dict = torch.load(filename)
        epoch = checkpoint_dict['epoch']
        model.load_state_dict(checkpoint_dict['model'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
        return epoch
    