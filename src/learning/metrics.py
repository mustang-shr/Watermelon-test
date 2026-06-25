"""
Formats the learning signal as an explicit before/after comparison, per the
brief: "Task X took 4 API calls on first run and 2 on the fifth run because
the agent learned Y" is the bar - a vague "it gets better" is explicitly
called out as not acceptable.

This is a formatting layer over MemoryStore.get_instruction_history_stats() -
the actual signal (call count, latency, per-instruction-cluster) is computed
and stored by the memory layer itself; this just makes it presentable.
"""


def learning_report(current_calls: int, current_time_ms: int,
                     similar_past: list[dict], history_stats: dict) -> str:
    if not similar_past or history_stats.get("count", 0) == 0:
        return (f"No prior similar instruction found - this run establishes the baseline "
                f"({current_calls} calls, {current_time_ms}ms).")

    avg_calls = history_stats["avg_api_calls"]
    avg_time = history_stats["avg_time_ms"]
    best_match = similar_past[0]

    call_delta = avg_calls - current_calls
    direction = "fewer" if call_delta > 0 else ("more" if call_delta < 0 else "the same number of")

    return (
        f"Matched against {history_stats['count']} similar past instruction(s) "
        f"(closest: '{best_match['text']}', similarity={best_match['similarity']:.2f}).\n"
        f"Past average: {avg_calls:.1f} API calls, {avg_time:.0f}ms.\n"
        f"This run: {current_calls} API calls, {current_time_ms}ms "
        f"({abs(call_delta):.1f} {direction} calls than the past average)."
    )
