from abc import ABC, abstractmethod

from slava.modules.agent.protection.models import ProtectionResult


class BaseProtector(ABC):

    @abstractmethod
    def check(self, query: str) -> ProtectionResult:
        """Check query"""
