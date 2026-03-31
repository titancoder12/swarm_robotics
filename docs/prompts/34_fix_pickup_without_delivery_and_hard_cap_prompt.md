You are working in an existing multi-agent swarm reinforcement learning codebase.

Task:
Fix the two major problems shown by the latest full MAPPO run:

1. training continues far past `--total-steps`
2. the policy still learns strong pickup behavior without learning delivery

Goal:
Make the MAPPO training path:

- respect `--total-steps` as a true hard global budget
- stop treating pickup-without-delivery as acceptable progress
- push full-swarm learning toward actual delivery conversion, not just raw harvest activity

Do not invent a new algorithm.
Do not solve this by simply increasing training budget.

--------------------------------------------------
PART 1 — CURRENT EVIDENCE
--------------------------------------------------

The latest run was launched with:

```bash
python train/train.py --backend mappo --headless --curriculum full --n-agents 6 --total-steps 600000 --rollout-steps 128 --update-epochs 4 --minibatch-size 256 --eval-every 10000 --eval-episodes 5 --reward-pickup 6 --reward-nest-delivery 30 --reward-undelivered-food -10 --folder-name mappo_full_600k_33
```

But the resulting artifacts show:

- run dir:
  - `runs/mappo_full_600k_33_20260330_220608`
- checkpoint root:
  - `checkpoints/mappo_full_600k_33`

Important evidence from `episode_metrics.csv`:

- `stage1a_single_agent_miniscule`
  - mean pickup `~= 3.74`
  - mean delivery `~= 3.74`
- `stage1d_single_agent_guaranteed_homing`
  - attempt 1 mean delivery `~= 0.486`
  - attempt 2 mean delivery `~= 0.594`
- `stage1e_single_agent_delivery_bridge`
  - attempt 1 mean delivery `~= 0.039`
  - attempt 2 mean delivery `~= 0.021`
- `stage1f_single_agent_delivery_obstacles`
  - mean delivery `0.0`
- `stage2a_small_swarm_medium`
  - attempt 1 mean pickup `~= 6.217`
  - attempt 1 mean delivery `0.0`
- `stage3a_full_swarm_large`
  - attempt 1 mean pickup `~= 13.522`
  - attempt 1 mean delivery `0.0`
- `stage3b_full_swarm_final`
  - attempt 1 mean pickup `~= 7.387`
  - attempt 1 mean delivery `0.0`

Important evidence from `eval_metrics.csv`:

- `stage1a` greedy delivery is strong
- `stage1b` and `stage1c` still have some greedy delivery
- from `stage1e` onward, greedy delivery is effectively zero
- full-swarm stages still show pickup with zero delivery

Important evidence from checkpoint metadata:

- `checkpoints/mappo_full_600k_33/latest/metadata.json`
  - `global_step = 1025006`
  - even though `--total-steps = 600000`

Interpretation:

- the current trainer is still not using `--total-steps` as a hard stop
- repeated stages and chunked rollout progression are letting the run overshoot badly
- the policy is still allowed to look “productive” through pickup and respawn activity, even when delivery remains zero

--------------------------------------------------
PART 2 — HARD GLOBAL TRAINING CAP
--------------------------------------------------

This must be fixed.

Requirements:

1. `--total-steps` must become a real global ceiling
2. the trainer must not continue deep into later stages once the global step budget is exhausted
3. stage repeats and rollouts must respect the hard stop

Preferred behavior:

- training may finish the current micro-step or minimal safe boundary
- but it must not overshoot by hundreds of thousands of steps

Review:

- stage loop control
- rollout loop control
- stage repeat interaction with global budget
- end-of-run checkpoint behavior when stopping on global budget

--------------------------------------------------
PART 3 — PICKUP-WITHOUT-DELIVERY FAILURE MODE
--------------------------------------------------

The run is still showing the same fundamental failure:

- high pickup
- zero delivery

This must no longer be treated as acceptable progress.

Requirements:

1. treat pickup-without-delivery as a failure mode, especially in later single-agent and swarm stages
2. make it much harder for a stage to look successful if delivery conversion is near zero
3. ensure full-swarm stages do not “pass” by harvesting alone

This applies to:

- stage scoring
- promotion logic
- best-checkpoint selection
- runtime summaries

--------------------------------------------------
PART 4 — DELIVERY-CONVERSION AS A PRIMARY SIGNAL
--------------------------------------------------

The trainer should care explicitly about:

- `food_retrieved`
- `delivery_conversion`

not just:

- `food_picked_up`
- reward

Required changes:

1. strengthen delivery-conversion importance in greedy scoring
2. make stage summaries clearly expose “pickup-rich, delivery-zero” failures
3. ensure best-checkpoint logic aligns with actual completed delivery

If needed, delivery conversion can become a first-class gating metric for later stages.

--------------------------------------------------
PART 5 — STAGE PROGRESSION FOR LATER STAGES
--------------------------------------------------

The later stages currently allow the run to move into large swarm settings even though delivery is still broken.

Required review:

- promotion targets for:
  - `stage1e`
  - `stage1f`
  - `stage2a`
  - `stage2b`
  - `stage3a`
  - `stage3b`

Preferred direction:

- later stages should require at least nontrivial greedy delivery
- zero-delivery greedy eval should not count as acceptable advancement

Do not make the targets unrealistic, but do make them meaningful.

--------------------------------------------------
PART 6 — PHEROMONE AND PICKUP-ONLY ACTIVITY
--------------------------------------------------

The current run shows pheromone deposition and respawn activity even while delivery remains zero.

That is dangerous because it allows the system to create the appearance of productive swarm behavior without actual completion.

Required review:

- whether pheromone-related behavior in later stages is helping true delivery
- whether pickup-only loops are being reinforced too much before delivery is stable

You may adjust stage-wise pheromone usage or reward emphasis if needed, but keep the trail-building goal intact.

The main principle is:

- trail behavior should support completed delivery, not replace it

--------------------------------------------------
PART 7 — DOCUMENTATION
--------------------------------------------------

Update docs as part of the implementation.

Required:

1. update `README.md`
2. update `docs/TRAINING_MAPPO.md`
3. update `docs/QandA.md`
4. update `docs/PROJECT_LOG.md`

Docs should explain:

1. why the old run exceeded `--total-steps`
2. how prompt 34 fixes the hard global stop
3. why pickup-without-delivery is now treated more strictly
4. what later-stage promotion now requires
5. which checkpoint should be demoed after the fix

--------------------------------------------------
PART 8 — VERIFICATION
--------------------------------------------------

After implementation:

1. run syntax/import checks
2. run at least one smoke run that proves:
   - the trainer does not wildly overshoot the requested total steps
   - later-stage summaries clearly expose pickup-vs-delivery conversion
   - zero-delivery greedy behavior is not treated as good progress
3. show exact commands used
4. explain remaining limitations honestly

--------------------------------------------------
PART 9 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. concise summary of the hard-stop fix
2. what changed in stage scoring / promotion
3. what changed in checkpoint selection
4. what the verification run showed
5. exact verification commands run
6. remaining limitations

--------------------------------------------------
IMPORTANT
--------------------------------------------------

The main objectives are now:

- **make `--total-steps` a true hard cap**
- **stop rewarding pickup-only behavior as if it were success**

The fix should be centered on:

- trainer control flow
- delivery-sensitive scoring and promotion
- later-stage discipline

not on adding more complexity to the environment.
