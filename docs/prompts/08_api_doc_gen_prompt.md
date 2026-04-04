Review the existing codebase and generate comprehensive developer-facing API documentation for the swarm robotics simulation environment.

Goal:
Produce clear, precise, implementation-grounded API documentation for developers who want to:
1. understand the environment,
2. inspect the observation and action spaces in detail,
3. write experiments and training scripts,
4. extend the simulator with new tasks, sensors, rewards, and dynamics.

Important:
- Do NOT write documentation based only on assumptions or the original design intent.
- Inspect the actual implementation in the repo and document what is truly implemented now.
- Where implementation differs from earlier design docs or comments, document the actual code behavior and note discrepancies.
- Be concrete and explicit. Avoid vague phrases like “normalized appropriately” unless you also state the exact formula/range/source in code.

Main deliverable:
Create a new file:

docs/API_REFERENCE.md

If helpful, you may also create:
- docs/OBSERVATION_SPEC.md
- docs/ACTION_SPEC.md
- docs/EXPERIMENT_GUIDE.md

But the main requirement is docs/API_REFERENCE.md.

Documentation requirements
==========================

1) Environment overview
Document:
- primary environment class(es)
- module/file locations
- how to instantiate the environment
- required config objects or parameters
- supported modes (render/headless/etc.)
- reset/step/render/close contract
- return types and shapes
- episode lifecycle
- task termination and truncation rules

2) Observation vector specification
This is the most important section.

For the observation vector, document ALL of the following:
- exact observation shape
- exact ordering of components in the vector
- semantic meaning of every component
- datatype of each component
- numeric range of each component
- unit of each component
- whether it is raw, normalized, clipped, binary, categorical, or one-hot
- exact normalization method/formula if normalized
- coordinate frame used:
  - world frame or agent/body frame
  - if relative vectors are used, relative to what
  - if heading-based transforms are used, explain them
- what happens when a value is unavailable
  - e.g. no visible target, no nearby neighbor, no pheromone present
  - does it become 0, max range, sentinel value, NaN, clipped value, etc.
- whether each feature is per-agent local observation or global/shared state
- whether any randomness/noise affects the observation
- which code function/method computes each observation component

I want this written as a developer spec, not just prose.

Please include a table like:

| Index | Name | Meaning | Source in code | Range | Unit | Normalization | Notes |

Examples of things I especially care about:
- lidar/ray distances:
  - how many rays
  - ray angles
  - max range
  - whether returned in pixels, meters, cells, or normalized fraction
  - whether wall and obstacle distances are combined
- target-relative features:
  - dx/dy or distance/bearing?
  - nearest visible target vs nearest overall target?
  - normalized how?
  - units?
- heading features:
  - raw theta vs sin/cos(theta)
  - angle units in code (radians or degrees)
- speed / velocity features:
  - scalar speed or vector velocity?
  - units per second, per step, pixels/sec, normalized?
- neighbor features:
  - nearest agent only or aggregated?
  - relative position in body frame or world frame?
  - units and normalization
- pheromone features:
  - exactly what samples are taken from the grid
  - how many values
  - what grid units mean
  - how concentrations are scaled/clipped/normalized

3) Action space specification
Also extremely important.

Document:
- exact action shape
- whether action space is discrete or continuous
- if discrete:
  - enumerate every action ID and what it means
  - include mapping table from action index -> behavior
- if continuous:
  - exact dimension
  - semantic meaning of each component
  - expected range
  - clipping behavior
  - unit / interpretation
- whether actions are high-level commands or direct motor commands
- how actions are transformed into movement by the dynamics driver
- whether different dynamics modes interpret the same action differently
- whether any smoothing, acceleration limits, turn-rate limits, or inertia are applied
- what happens for invalid actions

Please include a table like:

| Action / Index | Name | Meaning | Range | Unit | Internal interpretation | Notes |

4) Dynamics / state transition API
Document the internal state transition model:
- agent state variables actually used in code
- motion update order
- dt / timestep assumptions
- collision handling
- wall handling
- obstacle handling
- target collection rules
- pheromone deposit / decay / diffusion rules
- any stochasticity or domain randomization

If there are pluggable drivers (tank / hover / mixed), document:
- public interface
- key methods
- what state/action each expects
- how developers would add a new driver

5) Config API
Document the config/dataclass/schema in detail:
- every config field
- default value
- type
- valid range or expected options
- effect on simulation
- whether changing it affects obs_dim/action behavior
- any hidden coupling between config fields

Please include this as a proper parameter reference table.

6) Developer API for experiments
Document every public method/function/class a developer can use to build:
- experiments
- evaluation scripts
- random rollouts
- training loops
- benchmarks
- custom tasks

Examples:
- env constructors
- helper utilities
- rollout utilities
- training entry points
- benchmark configs
- evaluation functions
- rendering utilities
- seed control
- logging/stat collection
- replay buffer / policy interfaces if relevant

For each, document:
- function/class name
- file location
- purpose
- arguments
- return values
- side effects
- minimal usage example

7) Extension guide
Include a practical section:
- How to add a new observation component
- How to add a new reward term
- How to add a new action
- How to add a new dynamics mode
- How to add a new task/scenario
- How to add a new experiment

This should point to the exact files/classes/functions developers should modify.

8) Code-grounded examples
Add small examples for:
- creating env
- calling reset()
- stepping with one batch of actions
- inspecting obs shape
- decoding one observation row
- running a short random rollout
- switching config options
- enabling/disabling pheromones
- changing number of agents
- running headless evaluation

9) Accuracy requirements
Before writing the docs:
- inspect the code thoroughly
- trace how observations are built
- trace how actions are decoded
- trace how rewards, termination, and info are computed
- trace config usage
- trace experiment/training entrypoints

If any area is ambiguous:
- inspect more files until resolved
- if still unresolved, explicitly mark it as “unclear from current implementation” and cite the relevant code locations

10) Output quality
The docs should be:
- technical
- explicit
- implementation-accurate
- suitable for another developer to build on the codebase without reading every source file first

Please do not just summarize the project. I want a true API/specification document.

Suggested structure for docs/API_REFERENCE.md
=============================================
1. Environment Overview
2. Quick Start
3. Core Environment API
4. Observation Vector Specification
5. Action Space Specification
6. Rewards, Termination, and Info
7. Dynamics and State Transitions
8. Pheromone / Stigmergy System
9. Configuration Reference
10. Developer APIs for Training / Evaluation / Experiments
11. Extension Guide
12. Known Gaps / Implementation Notes

Final step:
After generating the documentation, also provide a short summary in your response of:
- which files you inspected,
- where the observation vector is defined,
- where the action mapping is defined,
- and any mismatches between implementation and earlier design intent.

Output:
Created docs/API_REFERENCE.md.

  What I inspected:

  - env/config.py
  - env/swarm_env.py
  - train/independent_dqn_pytorch.py
  - train/evaluate.py
  - train/experiment_utils.py
  - train/run_experiments.py
  - train/train.py
  - train/random_rollout.py
  - train/demo.py
  - experiments/benchmark_configs.py
  - analysis/plot_metrics.py
  - models/rule_based_policy.py
  - models/q_network.py

  Where the observation vector is defined:

  - shape and ordering: _compute_obs_dim() and _get_obs() in env/swarm_env.py
  - per-component helpers:
      - _lidar_scan()
      - _nearest_target_vector()
      - _nest_direction()
      - _nearest_agent_vector()
      - _food_presence()
      - _carrying_food()
      - _pheromone_samples()

  Where the action mapping is defined:

  - _build_action_table() in env/swarm_env.py

  Main implementation-vs-earlier-doc mismatches I noted:

  - current default observation is 23-D, not the older 19-D docs
  - current env includes nest-aware foraging and carrying-food state
  - render_scale exists in config but is unused in render
  - action_dim exists in config but actual action space uses num_actions
  - render(mode=...) accepts mode but ignores it
  - pheromone diffusion uses np.roll, so edges wrap around
  - pheromone_decay is implemented as multiplicative retention, not an explicit evaporation-rate subtraction model

  If you want, I can do a second pass and split the observation and action sections out into:

  - docs/OBSERVATION_SPEC.md
  - docs/ACTION_SPEC.md
    for easier reference.


Output 2:
    Added the split-out specs:

  - docs/OBSERVATION_SPEC.md
  - docs/ACTION_SPEC.md

  They are implementation-grounded extracts of the current contracts from:

  - _compute_obs_dim() and _get_obs() in env/swarm_env.py
  - _build_action_table() and the dynamics drivers in env/swarm_env.py

  So the current doc set for this area is now:

  - docs/API_REFERENCE.md
      - full developer API/spec
  - docs/OBSERVATION_SPEC.md
      - focused observation vector contract
  - docs/ACTION_SPEC.md
      - focused action-space contract

Output 3:
Added:

  - docs/CONFIG_REFERENCE.md
  - docs/EXPERIMENT_API.md

  So the focused reference set is now:

  - docs/API_REFERENCE.md
  - docs/OBSERVATION_SPEC.md
  - docs/ACTION_SPEC.md
  - docs/CONFIG_REFERENCE.md
  - docs/EXPERIMENT_API.md

  That gives you separate docs for:

  - full API
  - observation contract
  - action contract
  - config surface
  - experiment/evaluation framework