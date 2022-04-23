import numpy as np

class Dataset:
    """
    """
    def __init__(self,path:str=None) -> None:
        if path is not None:
            self.path = path
    def load(self) -> tuple(np.array):
        """
        """
        if self.path is not None:
            ...
        else:
            raise Exception(f'Dataset path Not defined')
    def transform(self,trans:list):
        """
        """
        ...