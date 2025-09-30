import numpy as np
import pandas as pd

class DataSet:
    def __init__(self, features, targets=None, binary_classification=False):
        """
        Dataset class.
        
        Args:
            - features (pd.Dataframe): the input features
            - targets (pd.DataFrame | np.ndarray): the targets (Optional)
            - binary_classification (bool): Whether we are consider it as a classification task
        """
        self.features = features.values
        if isinstance(targets, pd.DataFrame):
            self.targets = targets.values
        else:
            self.targets = targets

        if binary_classification and targets is not None:
            median = 4.364767
            self.targets = (self.targets > median).astype(int)
        
        self.size = len(self.features)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx], None
    
    def get_all(self):
        return self.features, self.targets

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        """
        DataLoader class.
        
        Args:
            - dataset (DataSet): The training/testing dataset
            batch_size (int): Num of data intances per batch
            shuffle (bool): If we need to shuffle the data during training/testing
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        self.current_idx = 0
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.dataset):
            raise StopIteration

        start = self.current_idx
        end = min(start + self.batch_size, len(self.dataset))
        batch_indices = self.indices[start:end]
        
        self.current_idx = end
        
        batch_x, batch_y = self.dataset[batch_indices]
        return batch_x, batch_y
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size