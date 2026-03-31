You are working in an existing multi-agent swarm reinforcement learning codebase.

Task:
Fix the current recurrent MAPPO training path so it stops collapsing into a low-motion / no-useful-action greedy policy, as observed in the `mappo_full_600k` run.

Goal:
The current run metrics show a serious problem:

- training episodes occasionally record pickup events
- but greedy evaluation remains near zero for pickup, delivery, and pheromone use across all stages
- demo checkpoints therefore appear frozen or nearly frozen

The objective is to make the MAPPO path learn behavior that survives into greedy evaluation and demo, not just stochastic training-time sampling.

Important evidence from the current repo state:

- `runs/mappo_full_600k_20260330_202140/eval_metrics.csv` shows:
  - `food_picked_up = 0`
  - `food_retrieved = 0`
  - `pheromone_deposit_events = 0`
  - across every evaluation stage
- the eval rewards are close to pure timeout step cost
- `train/demo.py --backend mappo` therefore shows agents that often do not move meaningfully

Do not solve this by inventing a new algorithm.
Fix the current MAPPO training path and curriculum behavior.

--------------------------------------------------
PART 1 — HIGHEST-PRIORITY DIAGNOSIS
--------------------------------------------------

Treat this primarily as a **policy-collapse / curriculum-stability** issue.

The current likely failure modes are:

1. the actor learns behavior that only works under stochastic action sampling during training, not under greedy argmax evaluation
2. critic resets across curriculum stages destabilize training
3. the curriculum advances on fixed step budgets even when the stage objective has not actually been learned
4. movement risk / penalties still overpower useful task acquisition in practice

You must address these directly in code and training structure.

--------------------------------------------------
PART 2 — CRITIC / STAGE-TRANSFER FIXES
--------------------------------------------------

This is the most important structural issue to investigate and fix.

Current issue:

- `train/mappo_gru.py` currently reinitializes actor and critic at each stage
- only actor weights are carried forward between stages
- critic reuse only happens on explicit resume when dimensions match
- the centralized critic input size currently changes with stage/environment shape

Required outcome:

1. reduce or remove critic-reset instability across curriculum stages
2. preserve as much training continuity as possible across stages
3. make the stage-transfer logic easy to inspect and explain

Preferred direction:

- move toward a fixed-dimension centralized critic input across stages if feasible
- if fixed critic input is too invasive for one pass, implement the strongest clean fallback that materially improves critic continuity

Do not ignore this issue.
This is the top structural problem.

--------------------------------------------------
PART 3 — CURRICULUM ADVANCEMENT QUALITY
--------------------------------------------------

The curriculum should not progress only because step budget elapsed.

Current issue:

- a stage can finish even if the policy never learned the intended behavior
- later stages then inherit a weak actor

Required improvements:

1. add stage-success-aware progression or at least stronger stage diagnostics
2. a stage should not silently “pass” if pickup/delivery remained effectively zero
3. preserve a practical CLI workflow

Preferred direction:

- support stage promotion based on minimum success criteria such as:
  - pickup rate
  - delivery rate
  - nonzero greedy eval success
- if fully automatic gating is too large for one pass, add at least a clean stop/repeat/manual-promote mechanism that is checkpoint-aware

--------------------------------------------------
PART 4 — GREEDY-EVAL ALIGNMENT
--------------------------------------------------

The current training path needs better alignment between what is learned during training and what is shown during demo.

Current issue:

- training samples actions from the policy distribution
- demo/eval use greedy argmax
- current behavior appears to rely on stochasticity and collapses under greedy execution

Required improvements:

1. strengthen the training signal toward robust greedy behavior
2. add clearer evaluation/selection around greedy performance
3. do not rely only on “latest checkpoint”

Preferred implementation ideas:

- save best checkpoints by greedy evaluation metrics, not only latest
- make greedy eval more central in training feedback
- consider whether entropy/exploration scheduling should decay more deliberately

The goal is:

- training-time success should transfer into greedy demo behavior

--------------------------------------------------
PART 5 — REWARD / CONTROL PRESSURE
--------------------------------------------------

Refine the current reward/control balance so useful motion is not suppressed.

Current issue:

- timeout-like negative rewards dominate eval
- collision risk and other negative pressure may still make movement unattractive in practice

Required outcome:

1. movement toward useful objectives should be easier to sustain
2. pickup and especially delivery should survive under greedy execution
3. changes should remain disciplined and documented

Preferred direction:

- review step cost / collision cost / undelivered penalty / delivery reward together
- if stage-wise tuning is helpful, use it cleanly
- be cautious about over-penalizing failed movement in early learning stages

Do not make the shaping flashy.
Make it more learnable.

--------------------------------------------------
PART 6 — CHECKPOINT SELECTION / DEMO USABILITY
--------------------------------------------------

Right now `latest/` is not enough for judging behavior quality.

Required improvements:

1. save checkpoints that are explicitly best according to meaningful eval criteria
2. make it obvious which checkpoint should be demoed
3. document the difference between:
   - latest
   - last stage
   - best greedy-eval checkpoint

Good criteria include:

- greedy eval delivery
- greedy eval pickup
- greedy eval reward

--------------------------------------------------
PART 7 — DOCUMENTATION REQUIREMENTS
--------------------------------------------------

Update docs as part of the implementation.

Required:

1. update `README.md`
2. update `docs/TRAINING_MAPPO.md`
3. update `docs/QandA.md`
4. update `docs/PROJECT_LOG.md`

Documentation must explain:

1. why the previous MAPPO path could look successful in training but fail in greedy demo
2. what was changed to reduce policy collapse
3. how stage transfer now works
4. how stage progression now works
5. which checkpoint users should demo
6. how to run training and demo with the improved path

--------------------------------------------------
PART 8 — VERIFICATION REQUIREMENTS
--------------------------------------------------

After implementation:

1. run syntax/import checks
2. run at least one short smoke training run
3. show that:
   - stage transfer still works
   - the new checkpoint-selection path works
   - greedy evaluation metrics are being recorded and used
4. show exact commands used
5. explain remaining limitations honestly

--------------------------------------------------
PART 9 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. concise summary of the anti-collapse changes
2. what was wrong with the old actor/critic stage-transfer logic
3. what changed in stage progression
4. what changed in greedy-eval checkpoint selection
5. what reward/control changes were made, if any
6. which checkpoint is now recommended for demo
7. exact verification commands run
8. remaining limitations

--------------------------------------------------
IMPORTANT
--------------------------------------------------

Prioritize in this order:

1. critic/stage-transfer stability
2. greedy-eval-aligned checkpoint quality
3. stage progression based on actual learning rather than only elapsed steps
4. reward/control tuning only as needed

Do not hide the problem with cosmetic demo changes.
Fix the training path so the learned behavior survives into greedy execution.
