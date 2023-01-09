import torch
import os
from torch.utils.data import Dataset

from pytorch_pretrained_bert import BertTokenizer

from utils import check_exists, makedir_exist_ok, save, load

import nltk
from datasets import load_dataset

class ELI5(Dataset):
    ''' 
        ELI5_Category (Long-Form QA) Dataset 
        Train: 91772
        Test: 5411
    '''
    def __init__(self, root, split) -> None:
        super().__init__()

        self.root = os.path.expanduser(root)

        if not check_exists(self.processed_folder):
            self.process()

        self.data = load(os.path.join(self.processed_folder, '{}_pre_tokenized.pickle'.format(split)), mode='pickle')

    def __getitem__(self, idx):
        input = {'Q': self.data['question'][idx], 'A': self.data['answer'][idx], 'Q_full': self.data['question_exp'][idx]}

        return input
        

    def __len__(self):
        return len(self.data['question'])
    

    def process(self):
        makedir_exist_ok(self.processed_folder)
        train_data, test_data = self.make_data('train'), self.make_data('test')

        save(train_data, os.path.join(self.processed_folder, 'train_pre_tokenized.pickle'), mode='pickle')
        save(test_data, os.path.join(self.processed_folder, 'test_pre_tokenized.pickle'), mode='pickle')

    def make_data(self, split):

        if split == 'train': 
            dataset = load_dataset("eli5_category", split='train') 
        elif split == 'test':
            dataset = load_dataset("eli5_category", split='validation1')

        """  Dataset: {features: ['q_id', 'title', 'selftext', 'category', 'subreddit', 'answers', 'title_urls', 'selftext_urls']} """

        question, question_exp, answer = [], [], []

        for i, ist in enumerate(dataset):
            question.append(nltk.word_tokenize(ist['title']))
            question_exp.append(nltk.word_tokenize(ist['selftext']))
            answer.append(nltk.word_tokenize(ist['answers']['text'][0]))

        return {'question': question, 
                'question_exp': question_exp, 
                'answer': answer}

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')




