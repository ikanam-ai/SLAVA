from typing import List

from slava.modules.agent.protection.base import BaseProtector
from slava.modules.agent.protection.models import ProtectionResult, ProtectionStatus


class ProtectorsAccumulator:
    def __init__(self, protectors: List[BaseProtector]) -> None:
        self.protectors = protectors

    def check(self, query: str) -> ProtectionResult:
        for protector in self.protectors:
            res = protector.check(query)
            if res.status is not ProtectionStatus.ok:
                return res
        return ProtectionResult(
            message="",
            status=ProtectionStatus.ok,
        )
