import enum
from dataclasses import dataclass
from typing import List

from slava.modules.agent.safe.protector_base import ProtectorBase


class ProtectionStatus(enum.Enum):
    ok = "ok"
    exceed = "exceed"


@dataclass
class ProtectionResult:
    message: str
    status: ProtectionStatus


class ProtectorAccumulator:
    def __init__(self, protectors: List[ProtectorBase]) -> None:
        self.protectors = protectors

    def validate(self, query: str) -> ProtectionResult:
        for protector in self.protectors:
            res = protector.validate(query)
            if res.status is not ProtectionStatus.ok:
                return res
        return ProtectionResult(
            message="",
            status=ProtectionStatus.ok,
        )
