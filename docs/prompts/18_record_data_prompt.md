You are working in an existing multi-agent swarm reinforcement learning codebase.

Task:
Implement a systematic swarm-scaling comparison workflow for three evaluation conditions using trained models:

1. a model trained WITH pheromone, evaluated WITH pheromone
2. a model trained WITHOUT pheromone, evaluated WITHOUT pheromone
3. the SAME model trained WITH pheromone, but evaluated with pheromone DISABLED

Goal:
I want to compare these three conditions across swarm sizes from 1 to 30 agents and generate clean CSV outputs and publication-quality matplotlib graphs.

The three comparison labels should be treated clearly throughout the code and outputs, for example:

- `trained_with_pheromone__eval_with_pheromone`
- `trained_without_pheromone__eval_without_pheromone`
- `trained_with_pheromone__eval_without_pheromone`

--------------------------------------------------
PART 1 — CLI ARGUMENTS
--------------------------------------------------

Add / update the evaluation entry point so it accepts explicit arguments for the required trained checkpoints:

1. checkpoint for model trained with pheromone
   Example:
   - `--checkpoint-with-pheromone path/to/model_with_pheromone.pt`

2. checkpoint for model trained without pheromone
   Example:
   - `--checkpoint-without-pheromone path/to/model_without_pheromone.pt`

3. optional general filename / experiment name
   Example:
   - `--filename pheromone_comparison_run`

4. swarm scaling arguments:
   - `--agent-min 1`
   - `--agent-max 30`
   - `--agent-step 1`
   - `--episodes-per-agent 10`

5. standard evaluation args if already supported:
   - `--output-dir experiment_data`
   - `--headless`
   - `--seed`
   - `--max-steps` or equivalent fixed evaluation horizon if already used in repo

Important:
- The script should NOT require a third checkpoint.
- The third condition must reuse the checkpoint provided by:
  - `--checkpoint-with-pheromone`
- but run evaluation with pheromone disabled.

--------------------------------------------------
PART 2 — THREE REQUIRED COMPARISON CONDITIONS
--------------------------------------------------

Implement evaluation for these three conditions:

Condition A:
- model trained WITH pheromone
- evaluate WITH pheromone enabled

Condition B:
- model trained WITHOUT pheromone
- evaluate WITH pheromone disabled

Condition C:
- model trained WITH pheromone
- evaluate WITH pheromone disabled

Important implementation rules:
- Condition C must use the same trained checkpoint as Condition A
- For pheromone-disabled evaluation:
  - no pheromone deposition
  - pheromone inputs/signals to the policy should be zeroed or disabled in a repo-consistent way
- Prefer keeping observation shape unchanged
- Do NOT break model input dimensionality
- If pheromone appears in the observation vector, return zeros for those channels/signals when disabled

For clarity, store metadata for each run including:
- comparison_label
- checkpoint_path
- trained_with_pheromone (bool)
- eval_with_pheromone (bool)

--------------------------------------------------
PART 3 — SWARM SCALING EVALUATION
--------------------------------------------------

Systematically evaluate swarm sizes:

- 1, 2, 3, ..., 30

Run multiple episodes per swarm size for each of the three comparison conditions.

Requirements:
- Use inference only
- Reuse existing evaluation utilities if possible
- Aggregate results across episodes
- Handle failures gracefully if one configuration fails

--------------------------------------------------
PART 4 — METRICS TO COLLECT
--------------------------------------------------

For every episode, collect at minimum:

- comparison_label
- checkpoint_path
- trained_with_pheromone
- eval_with_pheromone
- number_of_agents
- episode_index
- total_steps_taken
- targets_collected
- coverage
- coverage_efficiency
- efficiency
  - define as:
    targets_collected / number_of_agents
- time_to_first_discovery
  - first step at which any target is discovered/detected/picked up/collected
- total_reward
- pheromone_usage if available
- collisions if available
- filename / experiment_name
- seed if available
- episode_done_reason if available

Also include any existing repo metrics already available, such as:
- food_delivered
- targets_retrieved
- swarm_efficiency
- exploration_coverage
- new_cells_visited

Do not remove existing repo metrics.

Definitions:
- efficiency = targets_collected / number_of_agents
- coverage_efficiency = coverage / total_steps_taken
  or a repo-consistent equivalent if one already exists
- time_to_first_discovery should be clearly documented in code/comments

--------------------------------------------------
PART 5 — RAW CSV OUTPUT
--------------------------------------------------

Save all raw per-episode information to:

- `/experiment_data/raw/`

Create at least:

1. one master raw CSV containing all conditions
   Example:
   - `/experiment_data/raw/<filename>_all_conditions_raw.csv`

Each row should include:
- comparison_label
- checkpoint_path
- trained_with_pheromone
- eval_with_pheromone
- number_of_agents
- episode_index
- total_steps_taken
- targets_collected
- coverage
- coverage_efficiency
- efficiency
- time_to_first_discovery
- total_reward
- pheromone_usage
- collisions
- and all other available raw metrics

2. optionally, per-condition raw CSVs if convenient
   Example:
   - `/experiment_data/raw/<filename>_trained_with_eval_with_raw.csv`
   - `/experiment_data/raw/<filename>_trained_without_eval_without_raw.csv`
   - `/experiment_data/raw/<filename>_trained_with_eval_without_raw.csv`

Also create a summary CSV if practical:
- aggregated by condition and agent count

--------------------------------------------------
PART 6 — REQUIRED GRAPHS
--------------------------------------------------

Use matplotlib only.
Do not use seaborn.
Give each chart its own distinct figure.
Do not use subplots unless absolutely necessary.
Keep styling clean and readable.

Create the following required comparison line graphs:

1. Agents vs Targets Collected in Time
- line graph
- x-axis: number of agents
- y-axis: mean targets collected
- include all THREE conditions on the same plot
- use clear labels / legend
- include error bars or shaded variability if easy and repo-consistent

2. Coverage Efficiency vs Agents
- line graph
- x-axis: number of agents
- y-axis: mean coverage efficiency
- include all THREE conditions on the same plot

3. Efficiency vs Agents
- line graph
- x-axis: number of agents
- y-axis: mean efficiency
- efficiency = targets_collected / number_of_agents
- include all THREE conditions on the same plot

4. Time to First Discovery vs Agents
- line graph
- x-axis: number of agents
- y-axis: mean time to first discovery
- include all THREE conditions on the same plot
- handle no-discovery episodes cleanly

Important:
- These four comparison plots must each show all three conditions together
- Label the lines clearly and consistently

--------------------------------------------------
PART 7 — EXPLORATION VISUALS
--------------------------------------------------

Create exploration visuals for EACH of the three conditions separately.

Save them under:

- `/experiment_data/exploration_graphs/`

Requirements:
- produce separate exploration visuals for each comparison condition
- preferably create one or more representative exploration visualizations per condition
- use the simplest repo-consistent exploration visualization available:
  - visited-cell heatmap
  - occupancy map
  - visit-count heatmap
  - or binary visited/unvisited grid

Preferred:
- create visuals for representative swarm sizes such as:
  - 1, 5, 10, 20, 30

If that is too heavy, at minimum:
- create at least one representative exploration visual per condition

Suggested filename patterns:
- `/experiment_data/exploration_graphs/<filename>_trained_with_eval_with_agents_10.png`
- `/experiment_data/exploration_graphs/<filename>_trained_without_eval_without_agents_10.png`
- `/experiment_data/exploration_graphs/<filename>_trained_with_eval_without_agents_10.png`

If practical:
- also save PDF versions
But PNG is the priority for this directory unless the repo already cleanly supports both.

If possible, overlay:
- nest position
- target positions
- trajectories
This is optional, not required.

--------------------------------------------------
PART 8 — OUTPUT DIRECTORY STRUCTURE
--------------------------------------------------

Use this output structure:

- raw CSVs:
  - `/experiment_data/raw/`

- exploration visuals:
  - `/experiment_data/exploration_graphs/`

- comparison graphs:
  - save under an experiment_data graph directory, preferably:
    - `/experiment_data/graphs/PNG/`
    - `/experiment_data/graphs/PDF/`

Requirements:
- create directories automatically if they do not exist
- do not hardcode absolute filesystem paths
- interpret `/experiment_data/...` as relative to the chosen output root
- default output root should be:
  - `experiment_data`

For the comparison graphs, save both PNG and PDF if possible:
- `/experiment_data/graphs/PNG/<filename>_agents_vs_targets_collected.png`
- `/experiment_data/graphs/PDF/<filename>_agents_vs_targets_collected.pdf`
- etc.

Use:
- `plt.tight_layout()`
- readable labels, title, and legend
- descriptive filenames

--------------------------------------------------
PART 9 — IMPLEMENTATION RULES
--------------------------------------------------

- Reuse existing evaluation code where possible
- Do NOT rewrite the architecture
- Keep inference/evaluation runnable
- Prefer keeping observation dimensions unchanged
- When pheromone is disabled during evaluation:
  - zero pheromone inputs
  - disable deposition
  - keep network input shape compatible
- Do not add heavy dependencies

If the codebase already has a config object, add or reuse flags such as:
- `use_pheromone`
- `render_pheromone`
- `enable_pheromone_deposition`

Be practical and repo-aware.

--------------------------------------------------
PART 10 — SUMMARY / AGGREGATION
--------------------------------------------------

Aggregate results by:
- comparison condition
- number_of_agents

For each condition and agent count, compute at least:
- mean_targets_collected
- std_targets_collected
- mean_coverage_efficiency
- std_coverage_efficiency
- mean_efficiency
- std_efficiency
- mean_time_to_first_discovery
- std_time_to_first_discovery
- mean_total_reward
- std_total_reward

Use these summaries to generate the line graphs.

--------------------------------------------------
PART 11 — QUALITY REQUIREMENTS
--------------------------------------------------

- Real implementation only, not pseudocode
- Must work from command line
- Handle missing metrics gracefully
- Use deterministic seeds where practical
- If a single evaluation run fails, log it and continue if reasonable
- Keep naming clear and consistent
- Keep legend labels understandable
- Make graphs presentation-ready

--------------------------------------------------
PART 12 — DELIVERABLES
--------------------------------------------------

After implementing, provide:

1. list of modified / added files
2. example command to run the comparison
3. explanation of the three conditions
4. explanation of how pheromone-disabled evaluation is implemented
5. description of each generated CSV
6. description of each generated graph
7. description of exploration visuals
8. any assumptions made about discovery, targets collected, coverage efficiency, or no-discovery handling

--------------------------------------------------
EXAMPLE COMMAND
--------------------------------------------------

Example only; adapt to repo conventions:

python analysis/evaluate_comparison.py \
  --checkpoint-with-pheromone checkpoints/full_policy_with_pheromone.pt \
  --checkpoint-without-pheromone checkpoints/full_policy_without_pheromone.pt \
  --filename pheromone_comparison \
  --agent-min 1 \
  --agent-max 30 \
  --agent-step 1 \
  --episodes-per-agent 10 \
  --output-dir experiment_data \
  --headless

--------------------------------------------------
IMPORTANT
--------------------------------------------------

Implement the feature directly in the existing repo.
Keep it runnable.
Follow the project’s naming/style conventions.

Be practical and repo-aware:
- if “targets collected” is actually called “food delivered” or similar in this repo, adapt terminology accordingly
- if exploration visuals are not already supported, implement the simplest clean visited-cell heatmap possible using existing environment state
- prefer clear comparisons over over-engineering
- the main comparison is the difference between:
  1. trained with pheromone / eval with pheromone
  2. trained without pheromone / eval without pheromone
  3. trained with pheromone / eval without pheromone