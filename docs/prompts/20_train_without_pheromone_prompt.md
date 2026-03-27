# Train Without Pheromone Prompt
You are working in an existing multi-agent swarm reinforcement learning codebase for a stigmergic foraging environment.

Task:
Run a clean training job with pheromone disabled and save the checkpoint outputs under `checkpoints/without_pheremone`.

Requirements:
- Use the existing PyTorch DQN training entry point.
- Disable pheromone during training.
- Save checkpoints under `checkpoints/without_pheremone`.
- Use a clear filename so downstream evaluation can identify this as the pheromone-disabled model.
- Keep the command headless unless rendering is explicitly needed.

Run this command:

```bash
./.venv/bin/python train/independent_dqn_pytorch.py \
  --filename without_pheremone \
  --no-use-pheromone \
  --shared-policy \
  --headless \
  --total-steps 30000 \
  --save-dir checkpoints/without_pheremone \
  --save-every 5000
```

Expected outputs:
- `checkpoints/without_pheremone/1_4_trained/`
- `checkpoints/without_pheremone/1_2_trained/`
- `checkpoints/without_pheremone/3_4_trained/`
- `checkpoints/without_pheremone/full_policy/`
- `checkpoints/without_pheremone/metadata.json`

If you only want a smoke test first, use:

```bash
./.venv/bin/python train/independent_dqn_pytorch.py \
  --filename without_pheremone_smoke \
  --no-use-pheromone \
  --shared-policy \
  --headless \
  --total-steps 200 \
  --save-dir checkpoints/without_pheremone \
  --save-every 100 \
  --eval-every 0
```
