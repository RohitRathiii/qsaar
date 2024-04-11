import torch
from torch.utils.data import Dataset, DataLoader

class AyurBotDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def create_data_loader(texts, labels, tokenizer, batch_size=16):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
    dataset = AyurBotDataset(encodings, labels)
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader
