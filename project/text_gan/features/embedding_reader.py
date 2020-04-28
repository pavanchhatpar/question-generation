from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class EmbeddingReader(ABC):
    @abstractmethod
    def read(self,
             filename: str) -> Dict[str, np.ndarray]:
        pass
