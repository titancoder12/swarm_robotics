# RUN TRAINING FOR EXPERIMENTS
# With pheromone
./.venv/bin/python train/independent_dqn_pytorch.py \
  --use-pheromone \
  --shared-policy \
  --headless \
  --total-steps 5000000 \
  --save-dir checkpoints \
  --no-pheromone-requires-food \
  --filename with_pheromone_v2_5mil \
  --epsilon-start 1.0 \
  --epsilon-final 0.05 \
  --epsilon-decay-steps 300000 \
  --warmup-steps 20000 \
  --seed 42

# Without pheromone
./.venv/bin/python train/independent_dqn_pytorch.py \
  --no-use-pheromone \
  --shared-policy \
  --headless \
  --total-steps 5000000 \
  --save-dir checkpoints \
  --no-pheromone-requires-food \
  --filename without_pheromone_v2_5mil \
  --epsilon-start 1.0 \
  --epsilon-final 0.05 \
  --epsilon-decay-steps 300000 \
  --warmup-steps 20000 \
  --seed 42
