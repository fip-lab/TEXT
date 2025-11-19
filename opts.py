'''
* @name: opts.py
* @description: Hyperparameter configuration. Note: For hyperparameter settings, please refer to the appendix of the paper.
'''
import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    arguments = {
        'dataset': [
            dict(name='--datasetName',       
                 type=str,
                 default='mosi',
                 help='mosi, mosei or sims sims2'),

            dict(name='--dataPath',  
                 default="./datasets/mosi_unaligned_50_reason.pkl",
                 type=str,
                 help=' '),

            dict(name='--seq_lens',       
                 default=[50, 50, 50],
                 type=list,
                 help=' '),

            dict(name='--num_workers',  
                 default=8,
                 type=int,
                 help=' '),

           dict(name='--train_mode',         
                 default="regression",
                 type=str,
                 help=' '),

            dict(name='--test_checkpoint',    
                 default="./checkpoint/mosi/epoch_79_3layers.pth",
                 type=str,
                 help=' '),
        ],
        'network': [
            dict(name='--CUDA_VISIBLE_DEVICES',  
                 default='0',
                 type=str),

            dict(name='--fusion_layer_depth',     
                 default=2,
                 type=int)
        ],

        'common': [
            dict(name='--project_name',      
                 default='TEXT_Demo',
                 type=str),

           dict(name='--is_test',       
                 default=1,
                 type=int),
            dict(name='--seed',      
                 default=18,
                 type=int),

            dict(name='--models_save_root',  
                 default='./checkpoint',
                 type=str),

            dict(name='--batch_size',   
                 default=64,
                 type=int,
                 help=' '),

            dict(
                name='--n_threads',     
                default=3,
                type=int,
                help='Number of threads for multi-thread loading',),

            dict(name='--lr',      
                 type=float,
                 default=1e-4),

            dict(name='--weight_decay',      
                 type=float,
                 default=1e-4),

            dict(
                name='--n_epochs',      
                default=200,
                type=int,
                help='Number of total epochs to run',),

            dict(name='--use_bert',      
                 type=bool,
                 default=True),

            dict(name='--need_truncated',      
                 type=bool,
                 default=True),

            dict(name='--need_data_aligned',      
                 type=bool,
                 default=True),

            dict(name='--need_normalize',     
                     type=bool,
                     default=False),
        ],
        
        'global': [
            dict(name='--device', 
                 default=1,
                 type=int),

            dict(name='--epochs',  
                 default=100,
                 type=int),
            
            dict(name='--text_bert',  
                 default='text_bert',
                 type=str),
               
            dict(name='--note', 
                 default='',
                 type=str),

            dict(name='--dropout', 
                 default=0.0,
                 type=float),
               
            dict(name='--num_experts',  
                 default=5,
                 type=int),
               
            dict(name='--top_k',  
                 default=3,
                 type=int),
          
            dict(name='--capacity_factor',  
                 default=1.0,
                 type=float),

            dict(name='--result_txtPath',
                     default='./results.txt',
                     type=str,
                     help='Path to save the results'),
        ]
        
    }

    for group in arguments.values():    
        for argument in group:         
            name = argument['name']     
            del argument['name']     
            parser.add_argument(name, **argument)  

    args = parser.parse_args()     
    return args     

