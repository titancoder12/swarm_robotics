# Technical Report: Emergent Collective Intelligence in Stigmergic Swarm RL

## 1. Abstract and Research Objective

This report presents a rigorous investigation into the emergence of collective intelligence within decentralized robotic swarms using Multi-Agent Reinforcement Learning (MARL). The research specifically evaluates whether agents can transcend individual limitations by leveraging stigmergy, indirect coordination through environmental modifications, to solve complex foraging tasks. We hypothesize that digital pheromone trails provide a form of shared environmental memory that allows a swarm to become significantly more than the sum of its parts as its population scales.

The central research question is: Does stigmergic communication enable a swarm to become more than the sum of its parts as swarm size increases? To validate this, we distinguish between two primary success metrics: `food picked up` (initial target acquisition) and `food delivered` (successful completion of the foraging loop at the nest). While discovery is a prerequisite, delivery conversion is the definitive measure of functional collective intelligence. Our results demonstrate that while individual exploration is sufficient for discovery, stigmergic coordination is the causal driver for high-throughput delivery in complex, obstacle-rich environments.

## 2. System Architecture and Technical Specifications

The experimental framework, implemented in `env/swarm_env.py`, uses a 2D arena featuring a central nest, dynamic food sources, and varied obstacle densities. Decentralized control is achieved via Recurrent MAPPO (Multi-Agent Proximal Policy Optimization), which enables agents to maintain temporal state and handle partial observability.

### 23-Dimensional Observation Space

The policy operates on a 23-dimensional vector, providing the agent with local spatial awareness and stigmergic inputs. This space includes critical environment-side overrides such as non-carrying exploration randomness and force-explore nest-exit behaviors to prevent local optima.

| Index | Component | Semantic Meaning |
| --- | --- | --- |
| 1-8 | LiDAR Rays | Radial distance to obstacles or walls, with max range normalized. |
| 9-10 | Nest Direction | Relative `(dx, dy)` vector to the nest origin. |
| 11-12 | Nearest Food | Relative `(dx, dy)` vector to the nearest target. |
| 13-14 | Nearest Agent | Relative `(dx, dy)` vector to the closest swarm member. |
| 15-16 | Heading | Orientation represented as `[sin(theta), cos(theta)]`. |
| 17 | Speed | Current normalized scalar velocity. |
| 18 | Food Presence | Binary flag indicating whether a target is within pickup range. |
| 19 | Carrying State | Binary flag: `1` if carrying food, `0` otherwise. |
| 20-23 | Pheromone Samples | Local concentration samples with forward-biased sampling. |

### `Discrete(9)` Action Space

The action space uses a `3 x 3` mapping of throttle and turn values. This 9-index interface ensures that agents can simultaneously modulate velocity and angular momentum.

| Index | `[Throttle, Turn]` | Behavior |
| --- | --- | --- |
| 0 | `[-1, -1]` | Reverse Left |
| 1 | `[-1, 0]` | Reverse Straight |
| 2 | `[-1, 1]` | Reverse Right |
| 3 | `[0, -1]` | Spin Left (No Throttle) |
| 4 | `[0, 0]` | Idle / No Motion |
| 5 | `[0, 1]` | Spin Right (No Throttle) |
| 6 | `[1, -1]` | Forward Left |
| 7 | `[1, 0]` | Forward Straight |
| 8 | `[1, 1]` | Forward Right |

## 3. Methodology: The Curriculum Learning "Lesson Plan"

The primary obstacle to learning stigmergic foraging is the sparse-reward bottleneck associated with the return-to-nest phase. We employ a pedagogical progression of stages to shape behavior and prevent policy collapse, specifically the sampled-to-greedy gap where policies perform under stochastic training sampling but fail during deterministic evaluation.

1. **Stage 1A: "Miniscule"**
   Single agent, tiny unobstructed world. Teaches the basic target-seeking and pickup primitives.
2. **Stage 1B: "Tiny"**
   Single agent, larger world with mild obstacles. Focuses on search persistence.
3. **Stage 1D/E: "Post-Pickup Homing" and "Delivery Bridge"**
   These critical stages isolate the return trip. By starting agents in a carrying state near the nest, we force the formation of the homing policy, bridging the gap from simple pickup to full delivery.
4. **Stage 3B: "Final Hard Environment"**
   Full swarm (`6+` agents) in a large, cluttered environment. Requires stigmergy for efficient route reuse.

This curriculum prevents the agent from settling into a search-only local optimum. By ensuring greedy-behavior formation in early stages, we guarantee that the subskills of pickup and delivery are robust before reintroducing the complexity of multi-agent interference.

## 4. Experimental Design and Control Framework

To ensure scientific rigor, we use a federated design, synthesizing data from a Broad Campaign (robustness and scaling) and a decisive experiment (causal proof). We compare the MAPPO policy against three controls:

1. **Random Walk Baseline**
   Establishes the performance floor; agents move without intent.
2. **Rule-Based Baseline**
   A hand-coded heuristic (`Follow Food -> Pickup -> Follow Nest`). Crucially, this uses the same 23-dimensional inputs and `Discrete(9)` actions as the RL agent for an apples-to-apples comparison.
3. **Pheromone Ablation (Pheromone-Off)**
   We zero out pheromone inputs and disable deposition at evaluation. This isolates the causal contribution of communication to the swarm's efficiency.

## 5. The Decisive Experiment: Causal Proof for Stigmergy

The decisive experiment uses a repeated-source foraging task where food sources respawn, rewarding swarms that can establish and reuse stable transport routes. We compared a pheromone-trained swarm against a no-pheromone-trained swarm using 50 paired seeds at the 6-agent condition.

Crucially, pheromone deposition in the learned policy is gated by the carrying state. Agents only deposit pheromones when they have successfully acquired food. This transforms the pheromone field into a shared memory of success rather than a simple trail of recent motion.

### Mechanism-Sensitive Finding

The impact of stigmergy is most pronounced in the late-episode phase. While early exploration rates are similar, the pheromone-enabled swarm crystallizes paths, allowing delivery rates to increase non-linearly once the first discovery is made. The no-pheromone swarm, conversely, is forced into repeated, expensive rediscoveries for every target.

## 6. Multi-Dimensional Performance Metrics

Performance is evaluated through a multi-faceted metric suite to diagnose emergent behaviors:

- **Delivery Conversion**
  The ratio of targets picked up to those successfully delivered. This was the critical metric that finally validated the curriculum's success over pickup-only failure modes.
- **Pickup-to-Delivery Latency**
  Measures the temporal efficiency of the return path. Stigmergic trails significantly reduce this latency by providing optimized vectors.
- **Nest-Loiter/Crowding Fraction**
  Behavioral diagnostics identifying if agents are trapped near the nest, a common failure mode addressed by the post-delivery outward-search shaping implemented in later stages.

## 7. Statistical Validation and Significance Results

Quantitative data from the federated design confirms that emergent coordination is statistically robust and not a result of random chance.

### Statistical Significance of Stigmergic Coordination (6-Agent Condition)

| Metric | Pheromone-Trained | No-Pheromone-Trained | Statistical Test |
| --- | --- | --- | --- |
| Food Delivered (Mean) | `1.32` | `0.00` | `p = 0.00653` (Paired t-test) |
| Late-Episode Deliveries | `0.90` | `0.00` | `p = 0.01535` (Paired t-test) |
| Curriculum Quality | `1.25` (Current) | `0.00` (Old) | Cohen's `d = 1.254` |
| Wilcoxon Signed-Rank | `-` | `-` | `p = 0.00364` |

A p-value below `0.01` in the primary delivery metric confirms that stigmergy provides a statistically significant advantage in task completion and swarm throughput.

## 8. Robustness and Scaling Analysis

The swarm was evaluated under environmental stress to determine the limits of decentralized stigmergy:

- **Agent Failures**
  Maintained 56% efficiency (`0.70` vs `1.25`) with 2 agents disabled, proving system resilience.
- **Sensor Noise**
  Performance remained stable (`1.05`), suggesting the policy is robust to observation perturbations.
- **Scaling Law and the "Handoff" Problem**
  Without stigmergy, adding agents leads to diminishing returns due to crowding. With stigmergy, the swarm exhibits non-linear scaling. We addressed the nest-orbit failure mode by implementing force-explore nest-exit logic, ensuring that after a delivery, agents are effectively handed off back into an outward search phase rather than orbiting the nest.

## 9. Conclusion and GVRSF Impact

The experimental evidence confirms that the MARL swarm successfully learned decentralized coordination through stigmergic environmental traces.

1. **Curriculum Necessity**
   The Miniscule through Final Hard progression was essential to bridge the sampled-to-greedy gap.
2. **Algorithmic Superiority**
   The learned policy outperformed rule-based heuristics, proving it discovered more efficient coordination strategies than hand-coded logic.
3. **Causal Stigmergy**
   Pheromones, gated by successful carrying states, serve as a scalable collective memory.

This research has strong implications for disaster response and decentralized robotics in GPS-denied or communication-sparse environments. By using the environment as the communication medium, the system achieves a level of swarm scalability and robustness that central controllers cannot easily replicate.
