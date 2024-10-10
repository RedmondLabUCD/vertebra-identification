#import libraries 
import numpy as np
import pandas as pd
import time
import os
import itertools


# Functions to calculate averages of loss functions

class AverageBase(object):
    def __init__(self, value=0):
        self.value = float(value) if value is not None else None
       
    def __str__(self):
        return str(round(self.value, 4))
    
    def __repr__(self):
        return self.value
    
    def __format__(self, fmt):
        return self.value.__format__(fmt)
    
    def __float__(self):
        return self.value
    
class RunningAverage(AverageBase):
    """
    Keeps track of a cumulative moving average (CMA).
    """ 
    def __init__(self, value=0, count=0):
        super(RunningAverage, self).__init__(value)
        self.count = count
        
    def update(self, value):
        self.value = (self.value * self.count + float(value))
        self.count += 1
        self.value /= self.count
        return self.value

class MovingAverage(AverageBase):
    """
    An exponentially decaying moving average (EMA).
    """
    def __init__(self, alpha=0.99):
        super(MovingAverage, self).__init__(None)
        self.alpha = alpha
        
    def update(self, value):
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * float(value)
        return self.value
     
def monitor_progress(train_generator):
    start_time = time.time()
    for step, n_steps, loss in train_generator:
        elapsed = int(time.time() - start_time)
        print(f'\rBatch {step+1:03d}/{n_steps}  loss: {loss:0.6f}  elapsed: {elapsed}s',
              end='', flush=True)
    print()
    yield step, n_steps, loss  
        
def track_running_average_loss(train_generator):
    average_loss = MovingAverage()
    for step, n_steps, loss in train_generator:
        average_loss.update(loss)
        yield step, n_steps, average_loss.value
        
def run_train_generator(train_generator):
    for step, n_steps, loss in train_generator:
        pass
    return loss

