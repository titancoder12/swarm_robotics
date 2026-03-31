Implement the next curriculum fix for recurrent MAPPO.

Problem after prompt 40:
- The new small-swarm bootstrap stages are in place, but the focused `stage1_to_2` verification still spent too much of the total budget re-running already-solved single-agent return lessons.
- The run ended at `stage1f_single_agent_delivery_bridge` without cleanly reaching the new swarm bootstrap stack.
- The current failure mode is not “single-agent homing is impossible again”; it is “the curriculum is over-investing in already-solved stage-1 lessons before it reaches the new small-swarm stages.”

Goal:
- Preserve the prompt-39 single-agent return stack.
- Reduce unnecessary budget drain in `stage1d`, `stage1e`, and `stage1f`.
- Reach the new small-swarm bootstrap stages reliably in `stage1_to_2` verification and in full training.

Required implementation changes:

1. Reduce repeat pressure in the solved single-agent return stages.
- Lower repeat floors / repeat overrides for the single-agent carry/bootstrap/homing bridge stages so they do not consume too much of the total budget once they are “good enough”.
- Do not remove delivery-sensitive promotion entirely; just stop over-investing there.

2. Rebalance stage budgets slightly.
- Reduce the budget weights of the already-solved single-agent bootstrap stages if needed.
- Keep enough budget for them to remain stable, but shift more budget to the new swarm bootstrap stages.

3. Slightly relax the bootstrap-only promotion target where justified.
- The carry bootstrap stage (`stage1d`) does not need to be as strict as the later normal pickup-plus-delivery stages.
- If needed, make its promotion target easier than the bridge or obstacle-return stages, while keeping later stages delivery-first.

4. Keep docs current.
- Update README / MAPPO training docs / Q&A / project log to reflect the new stage-budget and repeat-pressure logic.

Verification:
- Run py_compile on touched files.
- Run a focused `--curriculum stage1_to_2` verification again.
- Confirm the verification reaches the new small-swarm bootstrap stages instead of exhausting the full budget inside repeated stage-1 bridge/carry lessons.
