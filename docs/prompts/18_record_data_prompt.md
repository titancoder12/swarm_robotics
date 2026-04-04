Work in the existing multi-agent swarm reinforcement learning codebase.

Task:
Extend the evaluation framework to include a random walk baseline and enforce that:

"targets_collected" refers to PICK-UP events, NOT delivery events.

Then implement a full swarm-scaling comparison across four conditions.

--------------------------------------------------
PART 1 — COMPARISON CONDITIONS
--------------------------------------------------

Implement evaluation for the following FOUR conditions:

1. trained_with_pheromone__eval_with_pheromone
   - model trained WITH pheromone
   - evaluated WITH pheromone enabled

2. trained_without_pheromone__eval_without_pheromone
   - model trained WITHOUT pheromone
   - evaluated WITHOUT pheromone

3. trained_with_pheromone__eval_without_pheromone
   - model trained WITH pheromone
   - evaluated WITHOUT pheromone

4. random_walk
   - NO model
   - NO learning
   - NO pheromone
   - random action selection at every step

Important:
- Condition 3 must reuse the checkpoint from condition 1
- Condition 4 must NOT use any checkpoint

--------------------------------------------------
PART 2 — CRITICAL METRIC DEFINITION
--------------------------------------------------

You MUST redefine:

targets_collected = number of PICK-UP events

NOT:
- delivery
- return-to-nest
- drop-off
- food_delivered

Implementation rules:

- Increment targets_collected the moment a target is first:
  - detected
  - picked up
  - collected

- If the repo currently uses:
  - food_delivered
  - targets_retrieved
  - resources_delivered

DO NOT use those unless they represent PICK-UP

If needed:
- add a new metric: targets_picked_up
- and map:
  targets_collected = targets_picked_up

Clearly document in code:
"targets_collected refers to pickup events, not delivery"

--------------------------------------------------
PART 3 — CLI ARGUMENTS
--------------------------------------------------

Add or update the evaluation entry point to support:

--checkpoint-with-pheromone
--checkpoint-without-pheromone
--filename
--agent-min
--agent-max
--agent-step
--episodes-per-agent
--output-dir
--headless
--seed
--max-steps

Important:
- Do NOT require a checkpoint for random_walk

--------------------------------------------------
PART 4 — RANDOM WALK BASELINE
--------------------------------------------------

Implement a random policy:

Requirements:
- No neural network
- No checkpoint
- Same action space as trained policy
- Works with multi-agent environment

Behavior:

At each step:
- each agent selects a random action from the action space

Example structure:

class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()

Multi-agent:
- each agent acts independently
- or vectorized sampling if supported

--------------------------------------------------
PART 5 — PHEROMONE CONTROL
--------------------------------------------------

For pheromone-disabled evaluation:

- disable pheromone deposition
- zero pheromone inputs in observation
- keep observation shape unchanged
- DO NOT break model compatibility

--------------------------------------------------
PART 6 — SWARM SCALING
--------------------------------------------------

Evaluate across:

agents = 1 → 30

For EACH condition:
- run multiple episodes
- use inference only
- aggregate results

--------------------------------------------------
PART 7 — METRICS TO COLLECT
--------------------------------------------------

Per episode, collect:

comparison_label
checkpoint_path
trained_with_pheromone
eval_with_pheromone
is_random_policy
number_of_agents
episode_index
total_steps_taken
targets_collected   (PICKUP-based)
coverage
coverage_efficiency
efficiency
time_to_first_discovery
total_reward
pheromone_usage
collisions
filename
seed
episode_done_reason

Also include existing repo metrics if available.

Definitions:

efficiency = targets_collected / number_of_agents
coverage_efficiency = coverage / total_steps_taken

time_to_first_discovery:
- first timestep where ANY agent picks up a target

--------------------------------------------------
PART 8 — RAW CSV OUTPUT
--------------------------------------------------

Save:

/experiment_data/raw/<filename>_all_conditions_raw.csv

Each row must include:
- condition label
- agent count
- episode index
- all metrics

Optionally:
- per-condition CSVs
- aggregated summary CSV

--------------------------------------------------
PART 9 — REQUIRED GRAPHS
--------------------------------------------------

Use matplotlib only.

Create 4 line graphs (ALL conditions included):

1. Agents vs Targets Collected
   - y = pickup-based targets_collected

2. Coverage Efficiency vs Agents

3. Efficiency vs Agents
   - targets_collected / number_of_agents

4. Time to First Discovery vs Agents

Requirements:
- all four conditions on same plot
- clear legend labels
- optional error bars

--------------------------------------------------
PART 10 — EXPLORATION VISUALS
--------------------------------------------------

Generate exploration visuals for EACH condition:

trained_with_pheromone__eval_with_pheromone
trained_without_pheromone__eval_without_pheromone
trained_with_pheromone__eval_without_pheromone
random_walk

Save under:

/experiment_data/exploration_graphs/

Preferred:
- heatmap of visited cells
- occupancy grid

Optional:
- trajectories
- nest/target overlays

--------------------------------------------------
PART 11 — OUTPUT STRUCTURE
--------------------------------------------------

experiment_data/
  raw/
  graphs/PNG/
  graphs/PDF/
  exploration_graphs/

Requirements:
- auto-create directories
- use relative paths
- save graphs as PNG and PDF if possible

--------------------------------------------------
PART 12 — SUMMARY AGGREGATION
--------------------------------------------------

Aggregate by:

condition + number_of_agents

Compute:

mean_targets_collected
std_targets_collected
mean_coverage_efficiency
std_coverage_efficiency
mean_efficiency
std_efficiency
mean_time_to_first_discovery
std_time_to_first_discovery
mean_total_reward
std_total_reward

--------------------------------------------------
PART 13 — QUALITY REQUIREMENTS
--------------------------------------------------

- real implementation only
- runnable from CLI
- deterministic seeds where possible
- handle failures gracefully
- do not break existing code
- reuse existing evaluation utilities

--------------------------------------------------
PART 14 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. list of modified / added files
2. example command to run
3. explanation of all four conditions
4. explanation of random_walk baseline
5. explanation of pickup-based targets_collected
6. description of CSV outputs
7. description of graphs
8. description of exploration visuals
9. assumptions made

--------------------------------------------------
EXAMPLE COMMAND
--------------------------------------------------

python analysis/evaluate_comparison.py \
  --checkpoint-with-pheromone checkpoints/full_policy_with_pheromone.pt \
  --checkpoint-without-pheromone checkpoints/full_policy_without_pheromone.pt \
  --filename pheromone_vs_random \
  --agent-min 1 \
  --agent-max 30 \
  --agent-step 1 \
  --episodes-per-agent 10 \
  --output-dir experiment_data \
  --headless
