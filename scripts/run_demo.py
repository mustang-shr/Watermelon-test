"""
Run with: python scripts/run_demo.py "your instruction here"
Or with no args, runs the three DEMO.md instructions in sequence.

Requires .env populated (see .env.example) and sourced into the environment
before running. This entry point is unverified end-to-end in this sandbox -
build.nvidia.com and write access to a real GitHub repo are both outside
what this environment can reach. Run and debug this locally.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import build_agent
from learning.metrics import learning_report

DEMO_INSTRUCTIONS = [
    "Create an issue titled 'Test issue from autonomous agent' with the body 'Created to verify the agent core works end to end.'",
    "Find all open issues with no assignee, group them by label, and post a summary comment on the most recently created one.",
    "Find duplicate issues among the open issues based on title similarity and report which ones look like duplicates.",
]


def run_one(agent, instruction: str):
    print(f"\n{'='*70}\nINSTRUCTION: {instruction}\n{'='*70}")
    result = agent.run(instruction)
    print(f"\nPlanner reasoning: {result.plan_reasoning}")
    if result.similar_past_used:
        print(f"\nMemory hit: {len(result.similar_past_used)} similar past instruction(s) found")
        for m in result.similar_past_used:
            print(f"  - '{m['text']}' (similarity={m['similarity']:.3f})")
    if result.synthesis_events:
        print(f"\nCapability synthesis triggered:")
        for ev in result.synthesis_events:
            print(f"  - {ev.tool_name}: success={ev.success}, attempts={ev.attempts}")
            if not ev.success:
                print(f"    last_error: {ev.last_error}")
    print(f"\n{result.report.summary()}")

    history_stats = agent.memory.get_instruction_history_stats(
        [m["id"] for m in result.similar_past_used]
    )
    print(f"\n--- Learning signal ---")
    print(learning_report(result.report.total_api_calls, result.report.total_time_ms,
                           result.similar_past_used, history_stats))
    return result


def main():
    agent = build_agent(use_real_embeddings=True)
    instructions = sys.argv[1:] if len(sys.argv) > 1 else DEMO_INSTRUCTIONS
    for instr in instructions:
        run_one(agent, instr)


if __name__ == "__main__":
    main()
