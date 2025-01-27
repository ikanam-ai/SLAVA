from slava.modules.agent.safe.protection_result import ProtectionResult, ProtectionStatus
from slava.modules.agent.safe.protector_base import ProtectorBase


class LengthLimitProtector(ProtectorBase):
    def __init__(self, max_len: int) -> None:
        self.max_len = max_len

    def validate(self, query: str) -> ProtectionResult:
        if len(query) > self.max_len:
            return ProtectionResult(
                message=f"Длина запроса превышает максимально допустимую величину в {self.max_len} символов.",
                status=ProtectionStatus.exceed,
            )
        else:
            return ProtectionResult(
                message="",
                status=ProtectionStatus.ok,
            )
