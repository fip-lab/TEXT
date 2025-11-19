'''
* @name: utils.py
* @description: Other functions.
'''


import os
import random
import numpy as np
import torch


class AverageMeter(object): 
    def __init__(self):   
        self.value = 0   
        self.value_avg = 0 
        self.value_sum = 0 
        self.count = 0 

    def reset(self):  
        self.value = 0      
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def update(self, value, count): 
        self.value = value
        self.value_sum += value * count  
        self.count += count            
        self.value_avg = self.value_sum / self.count  


def setup_seed(seed):
    torch.manual_seed(seed)          
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)            
    random.seed(seed)            
    torch.backends.cudnn.deterministic = True


def save_model(save_path, epoch, model, optimizer):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(save_path, 'epoch_{}.pth'.format(epoch))
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_file_path)


def best_result(test_results_list):
    best_results = {}
    for key in test_results_list[0].keys():
        best_value = max(
            enumerate(test_results_list), 
            key=lambda x: (x[1][key], x[0]) if key != 'MAE' else (-x[1][key], x[0])
        )[1] if key != 'MAE' else min(
            enumerate(test_results_list), 
            key=lambda x: (x[1][key], -x[0])
        )[1]

        best_epoch = test_results_list.index(best_value) + 1
        best_results[key] = (best_value[key], best_epoch)


    return best_results


def best_result_val(val_results_list, test_results_list):


    best_results = {}
    for key in val_results_list[0].keys():
        best_epoch_val = max(
            enumerate(val_results_list),
            key=lambda x: (x[1][key], x[0]) if key != 'MAE' else (-x[1][key], x[0])
        )[1] if key != 'MAE' else min(
            enumerate(val_results_list),
            key=lambda x: (x[1][key], -x[0])
        )[1]

        best_epoch = val_results_list.index(best_epoch_val) + 1
        best_value = test_results_list[best_epoch - 1][key]
        best_results[key] = (best_value, best_epoch)
    return best_results


