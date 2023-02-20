import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

class BasePolicy(ABC):
    def __init__(self):
        super().__init__()