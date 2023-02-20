from abc import ABC, abstractmethod

class BaseBid(ABC):
    def __init__(self,
                 id: str,
                 issuer: str,
                 amount: float,
                 price: float,
                 market: str,
                 type: str,
                 status: str,
                 confirmed_amount: float):

        super().__init__()

