from __future__ import annotations


class DoosanClearSkill:
    """Placeholder interface for future Doosan M0609 skill integration."""

    def clear_cup(self, cup_id: int) -> None:
        raise NotImplementedError("Connect this interface to the real Doosan CLEAR motion skill.")

    def spill_safe_clear_cup(self, cup_id: int) -> None:
        raise NotImplementedError("Connect this interface to the real Doosan SPILL_SAFE_CLEAR motion skill.")

