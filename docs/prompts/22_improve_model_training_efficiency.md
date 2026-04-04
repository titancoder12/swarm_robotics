Work in the existing multi-agent swarm reinforcement learning codebase.

Task:
Improve the training efficiency of the custom DQN implementation in `train/independent_dqn_pytorch.py` so it performs much closer to the SB3 path in `train/sb3_dqn.py`, while preserving current training behavior, checkpoint compatibility, and experiment outputs.

Goal:
I want a concrete implementation pass that removes the major performance bottlenecks in the custom trainer without breaking the current workflow for training, evaluation, logging, checkpointing, and analysis.

--------------------------------------------------
PART 1 — PERFORMANCE DIAGNOSIS TO ACT ON
--------------------------------------------------

The main reason SB3 is faster is that the custom trainer is doing too much Python-side per-agent work and too much extra bookkeeping inside the hot training loop.

From the current code, the concrete bottlenecks are:

1. Per-agent action selection loops
   - The trainer loops over agents one at a time.
   - For each agent it slices NumPy, creates a fresh torch tensor, unsqueezes, and runs a separate forward pass.
   - In `--shared-policy` mode this is especially inefficient because the same network is called once per agent instead of once on a batch.

2. Per-agent learning loops
   - After warmup, the trainer loops over every agent again.
   - It samples replay per agent, creates tensors per agent, runs forward/backward per agent, and does multiple optimizer steps.
   - In `--shared-policy` mode this should be consolidated into one batched update path.

3. Replay buffer tensor creation overhead
   - `ReplayBuffer.sample()` currently creates fresh `torch.tensor(...)` objects every sample.
   - That adds repeated allocations and copies before moving data to device.

4. Dict/array conversion overhead
   - The trainer repeatedly converts dicts to arrays and arrays back to dicts every environment step.
   - That overhead is not the biggest issue, but it adds up.

5. Heavy hot-loop bookkeeping
   - The training loop tracks many metrics, reward components, milestone checks, evaluation checks, and debug hooks every step.
   - This is useful, but too much work currently happens on the critical path.

6. Shared-policy mode is not truly batched
   - The code shares weights, but still keeps per-agent action selection and per-agent training loops.
   - The most important optimization is to batch compute properly in shared-policy mode.

--------------------------------------------------
PART 2 — REQUIRED CHANGES
--------------------------------------------------

Implement the following changes in `train/independent_dqn_pytorch.py`:

1. Batch action selection across agents in shared-policy mode
   - Replace the per-agent forward-pass loop with one batched forward pass over all current agent observations.
   - Use one `torch.as_tensor(obs, dtype=torch.float32, device=device)` call and one network call.
   - Preserve epsilon-greedy behavior exactly.
   - Preserve policy-debug support as much as practical.

2. Batch learning across agents in shared-policy mode
   - Preserve the current replay-buffer semantics unless a small redesign is clearly better.
   - In shared-policy mode, gather sampled minibatches across agent buffers, concatenate them into one larger batch, compute one Bellman target batch, one loss, and one optimizer step.
   - Do not keep one optimizer step per agent in shared-policy mode.

3. Reduce tensor allocation and copy overhead
   - Prefer `torch.as_tensor(...)` over `torch.tensor(...)` where safe and meaningful.
   - Avoid repeated small tensor creations in the hot path.
   - Keep tensor/device movement efficient and explicit.

4. Reduce avoidable Python overhead
   - Keep a fast internal array-based path where possible.
   - Minimize repeated dict/array conversion work without breaking the PettingZoo environment contract.

5. Lighten bookkeeping overhead without removing outputs
   - Keep episode metrics, CSV logging, checkpoints, milestone saves, and evaluation behavior.
   - But move or simplify work where possible so the hot loop does less Python-side overhead.
   - Preserve the current output schemas unless there is a very strong reason to change them.

6. Keep independent-policy mode working
   - The optimization focus is shared-policy mode.
   - Do not break the existing independent-per-agent mode.

--------------------------------------------------
PART 3 — WHAT MUST NOT BREAK
--------------------------------------------------

Preserve these behaviors:

1. Existing CLI arguments and training entrypoints
2. Existing checkpoint layout and filenames
3. Existing evaluation flow
4. Existing metric CSVs and summaries
5. Existing experiment runner expectations
6. Existing demo and policy-loading compatibility

Do not turn this into a full architecture rewrite unless absolutely necessary.
Prefer the smallest implementation that produces real speed gains.

--------------------------------------------------
PART 4 — IMPLEMENTATION GUIDANCE
--------------------------------------------------

Use these priorities:

Priority 1:
- Batch shared-policy inference across all agents

Priority 2:
- Batch shared-policy learning into one consolidated optimizer step

Priority 3:
- Reduce replay-buffer sampling overhead and unnecessary tensor creation

Priority 4:
- Trim hot-loop bookkeeping overhead where safe

Keep the code readable.
Add short comments only where the new batched logic is not immediately obvious.

--------------------------------------------------
PART 5 — VERIFICATION
--------------------------------------------------

After implementing the optimization:

1. Run at least a syntax or import-level check on the modified files
2. If feasible, run a short smoke test such as:

```bash
python train/independent_dqn_pytorch.py --headless --shared-policy --total-steps 200 --eval-every 0 --save-every 0
```

If that exact command needs adjustment because of current CLI constraints, use the nearest equivalent short smoke test and explain why.

--------------------------------------------------
PART 6 — DELIVERABLES
--------------------------------------------------

After the implementation, provide:

1. A concise summary of the concrete performance-oriented changes made
2. Which parts of the hot path are now batched
3. Any remaining bottlenecks that were intentionally left alone
4. Any behavior-preservation tradeoffs or assumptions
5. The exact verification commands run

--------------------------------------------------
IMPORTANT
--------------------------------------------------

The most important architectural fix is this:

When `--shared-policy` is enabled, stop treating each agent as a separate compute path for inference and learning. Batch them together.

Keep the repo runnable.
