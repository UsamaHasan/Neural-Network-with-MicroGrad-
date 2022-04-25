from abc import ABCMeta
from typing import Any

from numpy import isin

from utils.tensor import Tensor
class Module(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.grad = None
        self.trainable = True
        self.ctx = None
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(args[0])
    def forward(self,x):
        if isinstance(x,Tensor):
            x.ref = self
        self.ctx = x
    def backward(self):
        ...
    
from .activations import *
from .layer import *