from abc import ABC, abstractmethod
from typing import Iterable
from .landmark_set import LandmarkSet
import numpy as np


class ILandmarkDetector(ABC):
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> LandmarkSet:
        pass

    def detect_batch(self, images: Iterable[np.ndarray]) -> list[LandmarkSet]:
        return [self.detect(img) for img in images]

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass