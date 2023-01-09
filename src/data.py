from torch.utils.data import DataLoader
# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import BertModel

import torch
import numpy as np
from config import cfg
from dataset.eli5 import ELI5

def fetch_dataset(data_name):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = '{}/{}'.format(cfg['data_path'],data_name)

    if data_name in ['ELI5']:
        dataset['train'] = eval('{}(root=root, split=\'train\')'.format('ELI5'))
        dataset['test'] = eval('{}(root=root, split=\'test\')'.format('ELI5'))

    else:
        raise ValueError('Not valid dataset name')

    print('data ready')
    return dataset


def input_collate(batch):
    output = {key: [] for key in batch[0].keys()}
    for b in batch:
        for key in b:
            output[key].append(b[key])
    for key in output:
        if key != 'target_text':
            output[key] = torch.stack(output[key])
    return output 

def make_data_loader(dataset, tag, sampler=None):
    data_loader = {}
    for split in dataset:
        _batch_size = cfg[tag]['batch_size'][split] 
        _shuffle = cfg[tag]['shuffle'][split]
        
        if sampler is None:
            data_loader[split] = DataLoader(dataset=dataset[split], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[split] = DataLoader(dataset=dataset[split], batch_size=_batch_size, sampler=sampler[split],
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
    return data_loader


if __name__ == "__main__":
    dataset = fetch_dataset(cfg['data_name'])

    print(dataset['train'][1])