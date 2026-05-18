from __future__ import annotations


def wait() -> None:
    print("[MOCK ROBOT] WAIT")


def ask_user(cup_id: int, response_queue: list[str] | None = None) -> bool:
    if response_queue:
        response = response_queue.pop(0).strip().lower()
        print(f"Robot: Cup {cup_id} - 이 잔 치워드릴까요? (y/n)")
        print(f"User: {response}")
        return response.startswith("y")

    while True:
        response = input(f"Robot: Cup {cup_id} - 이 잔 치워드릴까요? (y/n)\nUser: ").strip().lower()
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("Please answer with y or n.")


def skip_cup(cup_id: int) -> None:
    print(f"[MOCK ROBOT] SKIP cup {cup_id}")


def approach_for_liquid_check(cup_id: int) -> None:
    print(f"[MOCK ROBOT] Approach cup {cup_id} for local liquid verification")


def clear_cup(cup_id: int) -> None:
    print(f"[MOCK ROBOT] CLEAR cup {cup_id}")


def spill_safe_clear_cup(cup_id: int) -> None:
    print(f"[MOCK ROBOT] SPILL_SAFE_CLEAR cup {cup_id} with slow upright motion")

