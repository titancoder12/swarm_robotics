# Train With Pheromone Prompt
You are working in an existing multi-agent swarm reinforcement learning codebase for a stigmergic foraging environment.

Task:
Run a clean training job with pheromone enabled and save the checkpoint outputs under `checkpoints/with_pheremone`.

Requirements:
- Use the existing PyTorch DQN training entry point.
- Enable pheromone during training.
- Save checkpoints under `checkpoints/with_pheremone`.
- Use a clear filename so downstream evaluation can identify this as the pheromone-enabled model.
- Keep the command headless unless rendering is explicitly needed.

Run this command:

```bash
./.venv/bin/python train/independent_dqn_pytorch.py \
  --filename with_pheremone \
  --use-pheromone \
  --shared-policy \
  --headless \
  --total-steps 30000 \
  --save-dir checkpoints/with_pheremone \
  --save-every 5000
```

Expected outputs:
- `checkpoints/with_pheremone/1_4_trained/`
- `checkpoints/with_pheremone/1_2_trained/`
- `checkpoints/with_pheremone/3_4_trained/`
- `checkpoints/with_pheremone/full_policy/`
- `checkpoints/with_pheremone/metadata.json`

If you only want a smoke test first, use:

```bash
./.venv/bin/python train/independent_dqn_pytorch.py \
  --filename with_pheremone_smoke \
  --use-pheromone \
  --shared-policy \
  --headless \
  --total-steps 200 \
  --save-dir checkpoints/with_pheremone \
  --save-every 100 \
  --eval-every 0
```
