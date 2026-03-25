# Add debug statements
You are working in an existing multi-agent swarm RL codebase.

Task:
Add debug logging so I can see, for each agent at each decision step, what the policy/model outputs and what final action the agent takes.

Goal:
I want to inspect the model’s behavior agent-by-agent during inference and evaluation, so I can understand whether the learned policy is behaving sensibly or if some agents are freezing, saturating on one action, or behaving inconsistently.

Requirements:
1. Find the code path where the trained policy is used to select an action for each agent.
   - This may be in evaluation code, rollout code, inference code, or inside the training loop during action selection.
   - Update the relevant path(s), especially the main inference/evaluation path.

2. Print a debug message for each agent that includes:
   - timestep / env step number
   - agent index or agent id
   - the raw model output before action selection
     - if DQN: print Q-values for all actions
     - if policy network: print logits / probabilities / continuous action outputs as appropriate
   - the selected action
   - whether the action came from exploration or greedy exploitation, if epsilon-greedy is used
   - optional but highly desirable: current epsilon value
   - optional but helpful: reward from previous step and done flag if available

3. The debug output must be easy to read.
   Use a consistent format like:

   [POLICY_DEBUG] step=123 agent=2 epsilon=0.142 mode=greedy q_values=[...] action=6
   or
   [POLICY_DEBUG] step=123 agent=2 logits=[...] probs=[...] action=left_forward

4. Make this behavior toggleable with a CLI flag or config option.
   Add something like:
   - `--debug-policy`
   - and optionally `--debug-policy-agents 0,2`
   - and optionally `--debug-policy-max-steps 100`

   So I can:
   - turn debugging on/off easily
   - restrict output to specific agents
   - avoid flooding the terminal for long runs

5. If the project has multiple entry points, wire this into the most useful ones:
   - evaluation / rollout script
   - inference script
   - optionally training script when running short debugging sessions

6. Keep the implementation minimal and clean.
   - Do not rewrite the architecture.
   - Do not change learning behavior unless necessary.
   - Do not introduce heavy dependencies.
   - Only add lightweight debug helpers and argument plumbing.

7. Add a small helper function if appropriate, for example:
   - `format_policy_debug(...)`
   - `should_debug_agent(...)`
   - `print_policy_debug(...)`

8. If actions are discrete and there is an existing action mapping table, include both:
   - action index
   - human-readable action meaning if available
   Example:
   `action=6 (forward_left)`

9. If the model output is a torch tensor, convert it safely for printing:
   - detach
   - move to cpu
   - convert to numpy or list
   - avoid breaking gradients in training code

10. If batch inference is used for all agents at once, still print one line per agent.

11. Protect output volume:
   - if `--debug-policy-max-steps` is provided, stop printing after that many env steps
   - do not print unless the debug flag is enabled

12. After making the changes, provide:
   - a short summary of which files were modified
   - exact example commands I can run
   - a short explanation of where the debug output appears

Implementation guidance:
- Search for where observations are passed into the model and where the chosen action is returned.
- If DQN is used, log:
  - obs shape if helpful
  - q_values
  - argmax action
  - exploration override if epsilon causes a random action
- If shared policy is used across agents, still print separate per-agent lines.
- If separate policies are used per agent, indicate which network is being used if practical.

Nice-to-have:
- Add optional compact formatting:
  - rounded values to 3 decimals
- Add optional summary at end of episode:
  - per-agent action counts
  - how often each agent chose each action

Deliverables:
- Modify the existing code directly.
- Keep the repo runnable.
- Do not give pseudocode only; implement the real changes.
- At the end, show:
  1. modified files
  2. what was added
  3. example commands to test it

Example expected usage:
- `python train/evaluate.py --checkpoint checkpoints/latest.pt --debug-policy`
- `python train/evaluate.py --checkpoint checkpoints/latest.pt --debug-policy --debug-policy-agents 0,1 --debug-policy-max-steps 50`

Important:
Make the changes repo-consistent and use the project’s existing naming/style conventions.