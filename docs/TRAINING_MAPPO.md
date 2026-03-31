# Recurrent MAPPO Training

This document describes the first runnable recurrent MAPPO path added to the
repo.

Source files:

- [train/mappo_gru.py](../train/mappo_gru.py)
- [algorithms/mappo/networks.py](../algorithms/mappo/networks.py)
- [algorithms/mappo/curriculum.py](../algorithms/mappo/curriculum.py)
- [algorithms/mappo/inference.py](../algorithms/mappo/inference.py)
- [docs/CTDE_STATE.md](CTDE_STATE.md)

## High-Level Design

The implementation uses:

- parameter-shared recurrent actor
- GRU actor over local observations
- recurrent centralized critic over a fixed padded training-time state dimension across the selected curriculum
- PPO-style clipped policy updates
- GAE
- entropy regularization
- gradient clipping

Inference remains decentralized:

- the actor consumes only local observations plus recurrent hidden state
- the critic is training-only

## Trail Objective

The current intended training behavior is not only “reach the target.”

The training path is now aligned to teach the full loop:

1. discover a target
2. pick it up
3. return to the nest
4. deposit pheromone on the successful return route
5. let later agents exploit that trail

This is reflected in the default reward ordering and pheromone behavior:

- pickup reward is meaningful but smaller than delivery reward
- delivery is the strongest task reward
- carrying-food progress back toward the nest gets a small signed shaping term
- pheromone following remains a small supportive signal
- pheromone deposition is gated so it is tied to carrying-food return behavior by default

The current default delivery mechanic is:

- agents may carry at most one food item at a time
- pickup sets an explicit carrying-food state
- reaching the nest while carrying counts as one completed delivery
- delivery clears the carrying state and returns the agent to its normal render color
- carrying agents render with a distinct green-highlighted body in demo mode

Food sources are now repeated-use sources instead of immediate single-use pickups:

- default total sources: `3`
- default source capacity: `4` uses each
- current semantics decrement capacity on pickup
- exhausted sources respawn elsewhere when target respawn is enabled

## Curriculum

The current implementation supports three curriculum modes:

- `stage1`
- `stage1_to_2`
- `full`

These modes now expand into explicit environment-difficulty stages instead of
only changing swarm size.

The default `full` schedule is:

1. `stage1a_single_agent_miniscule`
2. `stage1b_single_agent_tiny`
3. `stage1c_single_agent_small`
4. `stage1d_single_agent_return_medium`
5. `stage1e_single_agent_delivery_bridge`
6. `stage1f_single_agent_delivery_obstacles`
7. `stage2a_small_swarm_medium`
8. `stage2b_small_swarm_large`
9. `stage3a_full_swarm_large`
10. `stage3b_full_swarm_final`

The current curriculum stages the following environment variables:

- `n_agents`
- `width`
- `height`
- `n_targets`
- `n_obstacles`
- `max_steps`
- `active_targets`
- `target_respawn`
- `action_repeat_steps`
- `reward_new_cell`
- `reward_nest_approach`
- `carrying_reward_new_cell_scale`
- staged reward/pheromone simplifications for early greedy behavior
- stage-wise entropy start/end values used for within-stage entropy decay

Intended teaching progression:

- Stage 1A-1C: one agent, increasingly larger empty worlds with one target
- Stage 1D: one agent, medium world, one target, no obstacles; this is now the stricter homing stage and must show clearer greedy delivery before promotion
- Stage 1E: one agent, one target, one obstacle, no respawn; this is now a mild clutter bridge stage for carried return instead of the full obstacle challenge
- Stage 1F: one agent, one target, two obstacles, no respawn; this is the first true single-agent obstacle-return stage
- Stage 2A: small swarm, medium environment, two fixed sources, no respawn yet, with pheromone still disabled so early swarm delivery is learned before trail exploitation returns
- Stage 2B: small swarm, large but not final environment, now with respawn enabled
- Stage 3A: full swarm, same large but not final environment
- Stage 3B: full swarm, final large obstacle-heavy environment

The stage-budget bug fixed in the current version was that the old curriculum
reused early budget buckets and left later stages more starved than intended.
The schedule now assigns one explicit weight per actual stage:

- `1, 1, 1, 2, 2, 2, 2, 3, 4, 6`

This keeps the hardest full-swarm stages from receiving only accidental
fine-tuning time.

`--total-steps` now applies to the curriculum slice you actually selected. For
example, `--curriculum stage1 --total-steps 32000` distributes that full
`32000` budget across the six single-agent stages instead of first splitting
it across all ten full-schedule stages and then discarding the unused ones.

Control/reward staging now also changes with difficulty:

- early single-agent stages use `action_repeat_steps = 1` for more responsive control
- later swarm stages use `action_repeat_steps = 2` for smoother execution
- early stages keep a slightly stronger `reward_new_cell`
- later stages reduce `reward_new_cell` so delivery and trail reuse compete less with generic wandering
- stage 1 now disables pheromone entirely so pickup/return/delivery is learned before trail exploitation is introduced
- stage 1 softens `reward_step` and `reward_collision`, and strengthens pickup/delivery cues, so freezing is less attractive than useful movement
- prompts 30 and 31 suppress or strongly reduce exploration reward while carrying in the return-focused stages (`carrying_reward_new_cell_scale = 0.0` there), so after pickup the agent is not still being paid to wander
- `reward_nest_approach` is now staged explicitly, with stronger values in the return-focused single-agent stages than in the final full-swarm stages
- prompt 31 also makes the homing and early clutter-return stages more delivery-sensitive at promotion time, so weak return policies do not silently advance
- later stages progressively restore the full pheromone-enabled trail-building setting

Prompt 29 also changes entropy handling:

- entropy is no longer effectively one fixed pressure throughout a stage
- each curriculum stage now defines `entropy_start` and `entropy_end`
- the trainer linearly decays entropy regularization within the stage so early updates explore more and late updates in the same stage become more greedy
- this is meant to reduce the sampled-success / greedy-failure gap seen after prompt 28

Mode semantics:

- `stage1` runs all six single-agent stages
- `stage1_to_2` runs the six single-agent stages plus the two small-swarm stages
- `full` runs all ten stages

Actor and critic weights are now both carried across stages.

The current implementation avoids the old critic-reset instability by:

- computing one fixed critic input size for the selected curriculum
- padding each stage's centralized state into that fixed size
- keeping the same critic and optimizer state across stage boundaries

Stage progression is now greedy-eval-aware:

- each stage always runs a stage-end greedy evaluation
- each stage has a minimum pickup/delivery promotion target
- if the target is not met, the stage can repeat up to `--stage-repeat-limit` times
- if the limit is exceeded, training advances but records that the stage did not promote cleanly

## Example Commands

Single-agent smoke test:

```bash
python train/train.py --backend mappo --headless --curriculum stage1 --n-agents 6 --total-steps 2400 --rollout-steps 64 --update-epochs 2 --minibatch-size 128 --eval-every 0 --no-plots --n-targets 3 --active-targets 3 --food-source-capacity 4 --target-respawn --folder-name mappo_trail_smoke
```

Main trail-learning run:

```bash
python train/train.py --backend mappo --headless --curriculum full --n-agents 6 --total-steps 600000 --rollout-steps 128 --update-epochs 4 --minibatch-size 256 --eval-every 10000 --eval-episodes 5 --reward-pickup 6 --reward-nest-delivery 30 --reward-undelivered-food -10 --folder-name mappo_trail_full
```

Why this command is more realistic than the older shorter examples:

- `600k` total steps gives the eight-stage curriculum meaningful late-stage time
- delivery now dominates pickup more clearly
- the stronger undelivered-food penalty makes `picked up but never returned` less acceptable
- the later full-swarm stages are still hard enough that `180k` or `200k` often remains undertrained

Useful additional control:

```bash
python train/train.py --backend mappo --headless --curriculum full --n-agents 6 --total-steps 600000 --stage-repeat-limit 1 --folder-name mappo_trail_full
```

- `--stage-repeat-limit 1`
  - allows one additional attempt for a stage if stage-end greedy evaluation still fails the promotion target
  - this matters more now because early-stage greedy behavior, not just sampled success, is the promotion target

Resume from a checkpoint:

```bash
python train/train.py --backend mappo --headless --curriculum full --resume-checkpoint checkpoints/my_run/latest
```

Rendered MAPPO demo:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_trail_full/best_greedy_eval --max-steps 300
```

Checkpoint recommendation:

- `latest/`
  - most recently written checkpoint
- `stage3b_full_swarm_final/`
  - last explicit final-stage checkpoint
- `best_greedy_eval/`
  - recommended demo checkpoint because it is selected by greedy evaluation quality rather than recency

What prompt 28 fixed structurally:

- fixed padded critic state across the selected curriculum
- critic/optimizer carryover across stage boundaries
- stage-end greedy promotion checks
- `best_greedy_eval/` checkpoint saving

What still remained broken after prompt 28:

- early stages could still get sampled training-time pickup/delivery without learning a greedy policy that repeated that behavior in evaluation

What prompt 29 changes:

- stage 1 is intentionally simplified and de-pheromonized
- reward/control pressure is eased in stage 1 so movement and task completion dominate freezing
- entropy decays within a stage instead of staying fixed
- trainer runtime prints now call out sampled-vs-greedy gaps, making it obvious when lucky sampled behavior is not surviving into greedy eval

What still remained broken after prompt 29:

- pickup became learnable, but carrying-food return-to-nest completion still collapsed in the first obstacle stage and in early swarm stages

What prompt 30 changes:

- adds a dedicated single-agent return stage before the first obstacle-return stage
- stages `reward_nest_approach` explicitly
- suppresses exploration reward while carrying in the return-focused stages
- keeps stage 2A pheromone-free so early swarm delivery is learned before pheromone exploitation returns
- adds pickup-to-delivery conversion reporting to runtime/eval summaries

Headless MAPPO evaluation:

```bash
python analysis/evaluate.py --policy-kind mappo_gru --checkpoint-dir checkpoints/mappo_trail_full/latest --n-agents 6 --episodes 10 --headless --output-dir runs/eval --filename mappo_trail_full_eval --active-targets 3 --food-source-capacity 4
```

## Checkpoints

Each stage writes:

- `actor.pt`
- `critic.pt`
- `trainer.pt`
- `metadata.json`

There is also a rolling:

- `latest/`

And a greedy-eval-selected checkpoint:

- `best_greedy_eval/`

The deployment-facing actor loader is in
[algorithms/mappo/inference.py](../algorithms/mappo/inference.py).
