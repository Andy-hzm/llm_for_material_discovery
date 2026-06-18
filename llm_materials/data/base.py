from abc import ABC, abstractmethod
from typing import Optional


class BaseSource(ABC):
    """
    Every data source implements two methods:
      - fetch(n)         -> list of raw records (dicts)
      - clean(record)    -> cleaned record, or None to drop it
    """

    @abstractmethod
    def fetch(self, n: int) -> list:
        """Fetch up to n raw records from the source."""

    @abstractmethod
    def clean(self, record: dict) -> Optional[dict]:
        """
        Clean a single record.
        Return the cleaned record, or None to drop it.
        """
