import abc

from typing import Dict, Any
import abc

class PersonalizationVectorCreation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def run(self) -> Dict[Any, float]:
        pass