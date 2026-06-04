from abc import ABC, abstractmethod

# Abstract Base Class (ABC) for VLMs to ensure they have load and generate methods.
class BaseVLM(ABC):
    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def generate(self, prompt: list) -> str:
        pass