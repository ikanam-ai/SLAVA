from slava.modules.agent.protection.base import BaseProtector
from slava.modules.agent.protection.models import ProtectionResult, ProtectionStatus


class ExceedingProtector(BaseProtector):

    def __init__(self, max_len: int) -> None:
        self.max_len = max_len

    def check(self, query: str) -> ProtectionResult:
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
