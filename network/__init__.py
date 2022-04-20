from abc import ABCMeta
from typing import Any
class Module(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.grad = None
        self.trainable = True
        self.ctx = None
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(args[0])
    def forward(self):
        ...
    def backward(self):
        ...
    
from .activations import *
from .layer import *