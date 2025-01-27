import enum
from dataclasses import dataclass


class ProtectionStatus(enum.Enum):
    ok = "ok"
    exceed = "exceed"


@dataclass()
class ProtectionResult:
    message: str
    status: ProtectionStatus
