from abc import ABCMeta
class Module(metaclass=ABCMeta):
    def forward():
        ...
    def backward():
        ...
    
from .activations import *
from .layer import *