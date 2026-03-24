Before writing code, inspect the repository and determine:
1. the exact observation vector structure and ordering
2. the exact action space format (discrete or continuous)
3. how trained checkpoints are currently loaded
4. which model class should be used for inference

Then implement a very simple beginner-friendly script that isolates the trained model and lets a user manually feed observation vectors and inspect outputs.

Goal:
Create a small standalone “policy probing” script for this repo. It should help a developer understand what the trained policy does for specific observation vectors, and build intuition about how the model behaves.

Important intent:
This is NOT production infrastructure. It is a teaching and playground tool.
Favor clarity, readability, and ease of modification over abstraction or extensibility.

Core Requirements:
- Keep the code as simple and explicit as possible
- Prefer a single file
- Avoid unnecessary abstractions, inheritance, managers, registries, config systems, YAML, Hydra, callbacks, etc.
- Write code that a beginner can read top-to-bottom
- Add clear comments throughout

--------------------------------------------------
SCRIPT BEHAVIOR
--------------------------------------------------

The script should:

1. Load a trained model checkpoint
2. Put the model in evaluation/inference mode
3. Define several hand-written observation test cases near the top of the file
4. Allow selecting a test case via:
   - editing a variable in the file, and/or
   - a small argparse flag like: --case target_ahead
5. Convert the observation vector into the correct tensor shape
6. Run one forward pass through the model
7. Print results in a clear, human-readable way

--------------------------------------------------
OBSERVATION HANDLING
--------------------------------------------------

- Infer the real observation structure from the repo (do NOT invent it)
- Create a list:
  OBS_NAMES = [...]
  matching the true observation order

- When printing, show:
  name: value
  for each observation component

Example:
  front_lidar: 0.12
  left_lidar: 0.85
  target_dx: 0.90
  target_dy: -0.05

--------------------------------------------------
SAMPLE TEST CASES (VERY IMPORTANT)
--------------------------------------------------

Define several test cases directly in the script like:

TEST_CASES = {
  "target_ahead": [...],
  "wall_ahead": [...],
  "pheromone_left": [...],
  "open_space": [...]
}

For EACH test case:
- Add a short explanation in comments describing what the situation represents
- Also print that explanation when the case is selected

Example:
"target_ahead":
  # Target is directly in front, no obstacles nearby

These test cases should:
- use realistic values based on the actual observation format
- help build intuition about the policy behavior

--------------------------------------------------
ACTION OUTPUT + EXPLANATION (VERY IMPORTANT)
--------------------------------------------------

If DISCRETE actions:
- Print:
  - all Q-values or logits
  - chosen action index
  - human-readable action name

- Define:
  ACTION_NAMES = [...]
  using the real action space from the repo

- ALSO include a short explanation of what the action means, for example:
  "forward_right = move forward while turning right"

If CONTINUOUS actions:
- Print each output dimension with labels
- Example:
  forward_velocity: 0.72
  yaw_rate: -0.30

- ALSO print a short interpretation:
  e.g. "moving forward moderately while turning left"

--------------------------------------------------
OUTPUT INTERPRETATION (KEY FEATURE)
--------------------------------------------------

After printing raw outputs, include a short explanation block like:

"What this means:"
- Explain in plain language what the model is choosing to do
- Connect the action to the observation
- Example:
  "The model sees the target ahead and slightly to the right, so it chooses to move forward and turn right."

This explanation should be simple, not overly long, and based on the actual input and output.

--------------------------------------------------
IMPLEMENTATION CONSTRAINTS
--------------------------------------------------

- Reuse existing model-loading code where possible
- Do NOT import or run full training loops
- Keep dependencies minimal
- Handle basic errors (missing checkpoint, wrong vector length)
- Make the script runnable from command line
- Add a short usage comment at the top

--------------------------------------------------
DELIVERABLE
--------------------------------------------------

After implementing the script, provide a concise explanation of:
1. where the file was added
2. how to run it
3. how to modify or add new test cases
4. how observation vectors are structured
5. any assumptions made about checkpoint loading or action format

--------------------------------------------------
VERY IMPORTANT STYLE RULE
--------------------------------------------------

Favor explicit, beginner-friendly code over reusable abstractions.

This script should feel like a simple lab playground that someone can understand in 2 minutes and start modifying immediately.