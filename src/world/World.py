from abc import ABC, abstractmethod

class World(ABC):

    @abstractmethod
    def evaluate_individual(self, genotype) -> float:
        pass

    @abstractmethod
    def geno2pheno(self, genotype) -> object:
        pass
