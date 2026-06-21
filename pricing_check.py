# lightweight helper used only by run_all_tests.py
def calculate_cost_check(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    PRICING = {
        "gpt-4o": (2.50, 10.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "gpt-4o-mini": (0.15, 0.60),
    }
    ip, op = PRICING.get(model, (0.15, 0.60))
    return round((prompt_tokens / 1_000_000) * ip + (completion_tokens / 1_000_000) * op, 8)
