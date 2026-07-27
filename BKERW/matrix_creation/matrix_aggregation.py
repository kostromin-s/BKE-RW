import abc

class MatrixAggregation(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def run(self, chosen_policy):
        raise NotImplementedError