from abc import ABC, abstractmethod
from contextlib import AbstractContextManager

class Landmarker(AbstractContextManager, ABC):
    @abstractmethod
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
