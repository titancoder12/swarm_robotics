

## 1. Neural Network Architecture and I/O for the Deployed Model

**Question:** What is the deployed model architecture? How many layers? What is the input and output?

**Answer:** The deployed physical-robot actor came from:

- `checkpoints/mappo_g/latest`

Saved metadata:

- `obs_dim = 69`
- `action_dim = 18`
- `hidden_size = 128`

Actor structure:

1. `Linear(69 -> 128)`
2. `ReLU`
3. `GRU(128 -> 128)` with 1 recurrent layer
4. `Linear(128 -> 128)`
5. `ReLU`
6. `Linear(128 -> 18)`

Input:

- 69-value observation vector
- built from 3 stacked history frames of 23 values each

Output:

- 18 action logits
- deployment used `argmax`

## 2. Layer-to-Layer Connectivity Clarification

**Question:** Are the ReLU layers fully connected?

**Answer:** No.

- `Linear` layers were fully connected
- `ReLU` layers were elementwise
- each ReLU output only depended on the corresponding previous linear output

Specific points clarified:

- layer 1 -> 2: linear transform then elementwise ReLU
- layer 4 -> 5: linear transform then elementwise ReLU
- layer 5 -> 6: fully connected `Linear(128 -> 18)`

## 3. Are the Network Outputs Actions?

**Question:** The output of the NN are actions?

**Answer:** Not directly. The network output was a vector of 18 action logits.

- the selected action was the index of the largest logit at deployment time
- that discrete action ID then mapped to:
  - throttle
  - turn
  - deposit

## 4. Why Did This Feel Like DQN?

**Question:** Is this like DQN?

**Answer:** It looked similar at inference time but was not the same algorithm.

Similarity:

- both had 18 outputs
- both could use `argmax` to choose a discrete action

Difference:

- DQN outputs Q-values
- MAPPO actor outputs policy logits
- the MAPPO model was recurrent and trained with PPO plus a critic
- the DQN baseline in the repo was feedforward and value-based

## 5. Are We Using a Distribution?

**Question:** Are we using a distribution?

**Answer:** Yes in training, not really in deployment.

- the actor output logits defined a `Categorical(logits=...)` distribution
- training used sampling, log-probabilities, and entropy
- physical robot runtime used greedy `argmax`

## 6. Are the Training and Inference Architectures Different?

**Question:** Are the architectures for training and inference different?

**Answer:** The actor architecture was the same, but the full pipeline was different.

- training used:
  - actor
  - critic
  - PPO update logic
  - categorical sampling / entropy
- inference used:
  - actor only
  - recurrent hidden state
  - greedy `argmax` action selection

So the actor network was the same, but the training system and deployment system were not.

## 7. How Do the Actor and Critic Networks Interact?

**Question:** How do the two networks interact?

**Answer:** They interacted through PPO training rather than by feeding one directly into the other.

Interaction pattern:

1. actor received local observations and produced action logits
2. a categorical distribution was formed from those logits
3. actions were sampled during training
4. critic received the centralized state and predicted a scalar value
5. critic outputs were used to compute advantages and returns
6. actor was updated using PPO surrogate loss with those advantages
7. critic was updated against the returns

Key files:

- `algorithms/mappo/networks.py`
- `train/mappo_gru.py`

## 8. Why Does a Policy Learn a Distribution Only to Use `argmax` Later?

**Question:** Why does it learn a policy distribution only to use `argmax`?

**Answer:** Training and deployment had different goals.

- during training, PPO needed a categorical policy distribution for:
  - sampling
  - log-probabilities
  - entropy regularization
  - policy-ratio updates
- during deployment on the physical robot, the goal was stable deterministic control
- so the robot used `argmax(logits)` instead of sampling

Short version:

- stochastic policy for learning
- greedy mode for deployment

## 9. How Was Pheromone Dropping Trained in Curriculum Training?

**Question:** When doing curriculum training, how did I train the agent to drop pheromone?

**Answer:** The agent was trained to drop pheromone through the action space plus reward shaping, not through supervised labels.

- the action space always included a deposit bit
- early curriculum stages had `pheromone_enabled=False`
- pheromones were first enabled in later small-swarm bridge and route-reuse stages
- deposition was gated by:
  - carrying food
  - making nestward progress
- there was:
  - a small deposit cost
  - a small reward for following useful pheromone gradients
- the agent learned to deposit because doing so improved long-term delivery return under PPO

Key code areas referenced:

- `algorithms/mappo/curriculum.py`
- `env/swarm_env.py`
- `env/config.py`

## 10. Does the Robot's Radar Turn in the Simulator?

**Question:** In the simulator, does the robot's radar turn?

**Answer:** Not as an independently spinning sensor.

- the simulator uses a fixed fan of lidar rays attached to the robot body
- those rays rotate only because the robot heading `theta` changes
- each ray angle is computed as:
  - `agent.theta + offset`
- the default setup uses 9 rays over a front-facing 180 degree arc

Key code areas referenced:

- `env/swarm_env.py` in `_lidar_scan()`
- `env/config.py` for `lidar_rays`, `lidar_max_range`, and `lidar_step`

## 11. Does the Physical Robot's Radar Turn?

**Question:** What about for the physical robot?

**Answer:** Yes. The physical robot is treated as having a turning scan stream.

- the ESP32 sends `scan` samples with explicit `angle` values
- the Pi receives an unordered stream of angle-distance points
- that stream is then collapsed into the same fixed 9-ray front-arc representation used by the policy
- one helper, `read_full_sweep()`, collects points until an angle repeats, which indicates one sweep has wrapped around

So the physical robot does not feed raw simultaneous rays to the policy. It feeds a swept scan that is rebucketed into the policy's fixed lidar format.

Key code areas referenced:

- `firmware/run.py` in `bucketize_scan()`
- `firmware/ant.py` in `read_sensor_lines()` and `read_full_sweep()`

## 12. How Well Do the Simulator and Physical Robot Correspond?

**Question:** Do they correspond well? Does the model in simulator work in physical environment?

**Answer:** They correspond well enough for deployment, but not perfectly.

What matches:

- both end up giving the policy 9 front-arc range values
- both use the same observation structure expected by the deployed actor
- the physical runtime was explicitly designed to reshape real sensor data into the simulator-trained format

What differs:

- simulator lidar is idealized simultaneous ray casting
- physical lidar is a time-smeared sweep over angles, then rebucketed
- simulator target detection is geometric and exact
- physical target detection is camera-based and noisy
- simulator movement is clean
- physical movement includes delays, slippage, motor variance, and sensor dropouts

Conclusion:

- the simulator model is intended to run on the physical robot
- the deployment path is real and supported by the codebase
- but this is a sim-to-real approximation, not a perfect one-to-one sensing and actuation match

Key code areas and docs referenced:

- `firmware/run.py`
- `docs/SimToReal.md`
- `docs/CAMERA_SETUP.md`

## 13. What Was Done About the Straighter-Bias Robot Behavior?

**Question:** A while back, I was trying to make the robot go straighter instead of turning/spinning. What was actually done, and is the current version using that behavior?

**Answer:** The durable solution in the repo was not a separate straight-only trained neural network. The repo evidence points to runtime control overrides added in `firmware/run.py`.

What was added:

- a `control-mode` switch with:
  - `policy`
  - `heuristic`
  - `hybrid`
- a forward-biased heuristic action selector
- a hybrid override path that can replace the learned policy with the heuristic
- increasingly aggressive straight-ahead logic in later commits

What the heuristic currently does:

- it heavily prefers forward motion
- it sets `turn = 0.0` in the current checked-in implementation
- it only backs up for very close frontal obstacles
- if camera lock is strong, it still prefers going straight rather than steering eagerly

What the commit history shows:

- `b67b246`: added a bypass mode
- `97c0afa`: added a hybrid mode
- `e544fb7`: changed thresholds to favor straight-ahead motion when space is clear
- `77873b6`, `50b5893`, `71ccb5a`, `60a789d`, `e3a1c32`: progressively made the controller more forward-biased and less likely to let the learned policy reintroduce spin behavior

What the current default does:

- the current runtime still defaults to `--control-mode policy`
- so the straighter experimental behavior exists, but it is not the default path
- to use it, the runtime must be launched in `heuristic` or `hybrid` mode

Conclusion:

- the repo supports straighter runtime behavior through control-mode overrides
- the current default deployment does not automatically use those overrides
- the evidence supports "runtime heuristic/hybrid straightening" more strongly than "new straight-only trained model"

Key code areas referenced:

- `firmware/run.py`
- `git log -- firmware/run.py`

## 14. Why Can `runsim` Show a Top Action That Does Not Match Visible Behavior?

**Question:** In the simulator launched from `runsim`, why does the top action sometimes not seem to match what the agent actually does?

**Answer:** `runsim` uses greedy policy selection, but the environment can still alter or delay what is executed.

What `runsim` does:

- it launches `train/demo.py`
- it uses `--backend mappo`
- it points to `checkpoints/mappo_g/latest`
- it does not enable demo epsilon randomness explicitly

So the policy choice itself is greedy. However, the demo environment loads behavior-affecting settings from the checkpoint metadata, and those settings can change what actually happens after the top action is chosen.

Important reasons the visible behavior can differ:

1. `action_repeat_steps = 2`
- an earlier chosen action can be held for another tick
- so the newly displayed top action may not take effect immediately

2. `non_carrying_explore_random_action_prob = 0.15`
- empty agents can have their requested action randomly replaced by an exploration action

3. `non_carrying_force_explore_mode = true`
- empty agents near the nest can have their requested action overridden with an outward-moving action

4. deposit is still gated
- an action with deposit requested does not guarantee visible pheromone
- carrying-food and nest-progress conditions still apply

Conclusion:

- `runsim` does use greedy top-action selection
- but the environment can override, hold, or suppress parts of the requested behavior
- so the visible result can differ from the top action label even without robot hardware in the loop

Key code areas referenced:

- `runsim`
- `train/demo.py`
- `env/swarm_env.py`
- `checkpoints/mappo_g/latest/metadata.json`
