'''
* @name: dataset.py
* @description: Dataset loading functions. Note: The code source references MMSA (https://github.com/thuiar/MMSA/tree/master).
'''

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import time


# Configure logger
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)    

__all__ = ['MMDataLoader']

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
            'sims2': self.__init_sims2
        }
        DATA_MAP[args.datasetName]()

    def __init_mosi(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)
        if self.args.use_bert:
            self.text = data[self.mode][self.args.text_bert].astype(np.float32) 
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
     
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)





        '''
        train 1368 === dict_keys(['raw_text', 'text_bert', 'audio_lengths', 'vision_lengths', 'classification_labels', 'regression_labels', 'classification_labels_T', 'regression_labels_T', 'classification_labels_A', 
        'regression_labels_A', 'classification_labels_V', 'regression_labels_V', 'text', 'audio', 'vision', 'id', 

        'raw_text_en', 'reason', 'visual_clue', 'audio_clue', 'text_clue', 'clue_summary', 'text_en_bert', 'reason_bert', 'visual_clue_bert', 'audio_clue_bert', 'text_clue_bert', 'clue_summary_bert'])
        '''
        self.visual_clue_bert = data[self.mode]['visual_clue_bert'].astype(np.float32)
        self.audio_clue_bert = data[self.mode]['audio_clue_bert'].astype(np.float32)
        self.text_clue_bert = data[self.mode]['text_clue_bert'].astype(np.float32)
        self.clue_summary_bert = data[self.mode]['clue_summary_bert'].astype(np.float32)

        self.reason = data[self.mode]['reason'] 
        self.visual_clue = data[self.mode]['visual_clue']  
        self.audio_clue = data[self.mode]['audio_clue'] 
        self.text_clue = data[self.mode]['text_clue'] 
        self.clue_summary = data[self.mode]['clue_summary'] 

        # logger.info(f"{self.mode} data loaded, shape of text befor truncat: {self.args.text_bert}: {self.text.shape}, audio: {self.audio.shape}, vision: {self.vision.shape}, audio_clue: {self.audio_clue_bert.shape}, visual_clue: {self.visual_clue_bert.shape}")

        if 'sims' in self.args.datasetName:
            self.rawText_en = data[self.mode]['raw_text_en']  

        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']
        self.labels = {
            'M': data[self.mode][self.args.train_mode+'_labels'].astype(np.float32) 
        }
        if self.args.datasetName == 'sims' or self.args.datasetName == 'sims2':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m].astype(np.float32)

        if not self.args.need_data_aligned:   
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']

        # clear dirty data    
        self.audio[self.audio == -np.inf] = 0
        self.vision[self.vision == -np.inf] = 0

        if self.args.need_truncated:
            self.__truncated()
        
        if self.args.need_normalize:
            self.__normalize()
        
        logger.info(f"{self.mode} data loaded, shape of text after truncat: {self.args.text_bert}: {self.text.shape}, audio: {self.audio.shape}, vision: {self.vision.shape}, audio_clue: {self.audio_clue_bert.shape}, visual_clue: {self.visual_clue_bert.shape}")


    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()
    
    def __init_sims2(self):
        return self.__init_mosi()

    def __truncated(self):
        print("Truncating data...")
        # NOTE: Here for dataset we manually cut the input into specific length.
        # NOTE 这里我们手动将输入切割成特定长度。
        def TruncatedText(text_features, length):
            # text_features 的形状为 (batch_size, 3, seq_len)
            if text_features.shape[2] > length:
                return text_features[:, :, :length]
            # 如果长度不足，进行零填充
            padding = np.zeros((text_features.shape[0], text_features.shape[1], length - text_features.shape[2]))
            return np.concatenate((text_features, padding), axis=2)
        
        def Truncated(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+length])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+length])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
                       
        text_length, audio_length, video_length = self.args.seq_lens

        self.vision = Truncated(self.vision, video_length)
        self.text = TruncatedText(self.text, text_length)  
        self.audio = Truncated(self.audio, audio_length)

        # clue
        self.audio_clue_bert = TruncatedText(self.audio_clue_bert, 50)
        self.visual_clue_bert = TruncatedText(self.visual_clue_bert, 50)
        self.clue_summary_bert = TruncatedText(self.clue_summary_bert, 50)



    def __normalize(self):
        print("Normalizing data...")
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # For visual and audio modality, we average across time:
        # The original data has shape (max_len, num_examples, feature_dim)
        # After averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))


    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.args.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.rawText[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
            
            'reason': self.reason[index],  
            'visual_clue': self.visual_clue[index],  
            'audio_clue': self.audio_clue[index], 
            'text_clue': self.text_clue[index], 
            'clue_summary': self.clue_summary[index], 

            'visual_clue_bert': torch.Tensor(self.visual_clue_bert[index]), 
            'audio_clue_bert': torch.Tensor(self.audio_clue_bert[index]), 
            'text_clue_bert': torch.Tensor(self.text_clue_bert[index]), 
            'clue_summary_bert': torch.Tensor(self.clue_summary_bert[index]), 
        } 
        if not self.args.need_data_aligned:  
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        return sample


def MMDataLoader(args):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len()  

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader