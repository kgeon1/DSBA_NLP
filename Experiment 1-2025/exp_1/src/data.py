from transformers import AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset, concatenate_datasets

import random
import torch
from torch.utils.data import Dataset, DataLoader

import omegaconf
from typing import Union, List, Tuple, Literal


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, data_config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test']):
        super().__init__()
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(data_config.model_name)
        self.max_len = data_config.max_len

        # IMDB 데이터 로드 및 병합
        raw_train_dataset = load_dataset("imdb", split='train')
        raw_test_dataset = load_dataset("imdb", split='test')
        full_dataset = concatenate_datasets([raw_train_dataset, raw_test_dataset])

        # train/valid/test split (9:1:1)
        full_indices = list(range(len(full_dataset)))
        random.seed(42)
        random.shuffle(full_indices)

        total_len = len(full_dataset)
        num_test_samples = int(total_len * 0.1)
        num_valid_samples = int(total_len * 0.1)

        test_indices = full_indices[:num_test_samples]
        valid_indices = full_indices[num_test_samples : num_test_samples + num_valid_samples]
        train_indices = full_indices[num_test_samples + num_valid_samples:]

        if self.split == 'train':
            self.data = full_dataset.select(train_indices)
        elif self.split == 'valid':
            self.data = full_dataset.select(valid_indices)
        else:
            self.data = full_dataset.select(test_indices)

        # 토큰화
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], 
                                  truncation=True, 
                                  padding="max_length", 
                                  max_length=self.max_len)

        self.tokenized_data = self.data.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )

        # 사용할 column 정의
        self.has_token_type_ids = 'token_type_ids' in self.tokenized_data.column_names
        cols = ['input_ids', 'attention_mask', 'label']
        if self.has_token_type_ids:
            cols.append('token_type_ids')

        self.tokenized_data.set_format(type="torch", columns=cols)

        print(f">> SPLIT : {self.split} | Total Data Length : {len(self.tokenized_data)} | token_type_ids: {self.has_token_type_ids}")

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx) -> Tuple[dict, int]:
        item = self.tokenized_data[idx]

        inputs = {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask']
        }
        if self.has_token_type_ids:
            inputs['token_type_ids'] = item['token_type_ids']

        label = item['label']
        return inputs, label

    @staticmethod
    def collate_fn(batch : List[Tuple[dict, int]]) -> dict:
        input_ids_list = [item[0]['input_ids'] for item in batch]
        attention_mask_list = [item[0]['attention_mask'] for item in batch]
        label_list = [item[1] for item in batch]

        data_dict = {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'labels': torch.tensor(label_list, dtype=torch.long)
        }

        # token_type_ids가 있는 경우만 포함
        if 'token_type_ids' in batch[0][0]:
            token_type_ids_list = [item[0]['token_type_ids'] for item in batch]
            data_dict['token_type_ids'] = torch.stack(token_type_ids_list)

        return data_dict


def get_dataloader(data_config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test']) -> torch.utils.data.DataLoader:
    dataset = IMDBDataset(data_config, split)
    dataloader = DataLoader(dataset, 
                            batch_size=data_config.batch_size, 
                            shuffle=(split=='train'), 
                            collate_fn=IMDBDataset.collate_fn)
    return dataloader
