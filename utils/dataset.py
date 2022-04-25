import random
from utils import split
import numpy as np
class Batch_scheduler:
    """
    """
    def __init__(self,x,y,batch_size=32,shuffle=True) -> None:
        """
        """
        self.x = x
        self.y = y
        assert len(self.x) == len(self.y)
        self.shuffle = shuffle
        self.batch_size = batch_size
        idx = [i for i in range(len(self.x))]
        self.idx = list(split(idx,batch_size))
        self.len = len(self.idx)
        self.ptr = 0
        if shuffle:
            random.shuffle(self.idx)
        
    def __getitem__(self,index) -> tuple[np.array,np.array]:
        x = self.x[index]  
        y = self.y[index]
        return x , y

    def __len__(self):
        return len(self.x[0]) 
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.ptr < self.len:
            idx = self.idx[self.ptr]
            self.ptr+=1
            return self[idx]
        else:
            self.ptr = 0
            raise StopIteration