# GVRSF Board Draft Optimized

## Title

**Stigmergy Is All You Need**

**Learning decentralized swarm intelligence through environmental communication**

Short subtitle:

**A multi-agent reinforcement learning swarm learns to explore, find targets, and return them to a nest without a central controller.**

Suggested visual:

- one large hero screenshot of the swarm with pheromone heatmap visible

---

## Motivation

**Problem:** Most robot systems rely on central control.\
**Question:** Can intelligence emerge without it?\
**Idea:** Use stigmergy — agents coordinate by changing the environment.

Optional 1-line origin story:

> Inspired by observing carpenter ants exploring and coordinating without a leader.

---

## Research Question

> **Does stigmergic communication enable a decentralized swarm to scale better with size?**

---

## Hypothesis

- Pheromone > no pheromone
- The performance gap increases with swarm size
- The strongest effect appears after route discovery
- Curriculum learning is needed for reliable delivery behavior

---

## Why This Is Computer Science

This project is not just a robotics demo. It studies:

- decentralized algorithms
- multi-agent reinforcement learning
- curriculum learning
- emergent behavior
- indirect communication through the environment

---

## What Is New in This Project

- RL + stigmergy integrated in one swarm system
- a staged curriculum that turns pickup behavior into delivery behavior
- a controlled scaling experiment testing whether stigmergy is causally important

---

## Learning Principle (Depth Anchor)

### Bellman Idea

**V(s) = E[r + γV(s')]**

Good actions improve future outcomes, not just immediate reward.

Why this matters here:

- exploration pays off later
- pheromones help future agents
- returning to the nest is delayed reward

**This is why stigmergy works.**

---

## Theory in Code

```python
delta = r + gamma * V(next) - V(current)
```

This is the value-learning signal that helps the policy learn long-horizon behavior.

---

## System Overview

### Task Loop

**explore → detect → pick up → return → deliver**

### System Diagram

```text
Observation → Policy → Action
        ↓
 Environment (pheromone)
        ↓
Next Observation
```

Add these labels in the final visual:

- nest
- food
- pheromone trails

Board goal:

- judges understand the system in 5 seconds

---

## Curriculum Learning

### Problem

Direct training was unstable.

### Solution

Train the swarm in stages, from easy subskills to full swarm coordination.

### Result

**0.00 → 1.25 deliveries/episode**

Board takeaway: **The curriculum was a major algorithmic contribution.**

Suggested visual:

- compact ladder from single-agent stages to full-swarm stages

---

## Experimental Design

### Independent Variables

- swarm size
- pheromone enabled vs disabled

### Dependent Variables

- food delivered
- late deliveries
- post-discovery deliveries

### Controls

- same seeds
- same environment geometry
- deterministic greedy evaluation

**Controlled comparison: only pheromone differs.**

---

## Results

### ⭐ Key Result: Stigmergy Enables Scaling

At **6 agents**:

- **With pheromone:** 1.32 deliveries
- **Without pheromone:** 0.00 deliveries

**p < 0.01**

Conclusion: **Coordination emerges only with stigmergy.**

Suggested graph:

- killer experiment scaling graph

---

### Supporting Result 1: Curriculum Learning Worked

- current checkpoint: **1.25 deliveries/episode**
- weaker checkpoint: **0.00**

Conclusion: **The staged training process was necessary to get real delivery behavior.**

Suggested graph:

- curriculum vs weaker training

---

### Supporting Result 2: Learned Swarm Beat Baselines

- MAPPO: **1.25**
- Rule-based: **0.30**
- Random: **0.05**

Conclusion: **The behavior was learned, not hand-scripted.**

Suggested graph:

- baseline comparison

---

## Why the Key Result Is Strong

The strongest pheromone effects appeared in:

- late deliveries
- post-discovery deliveries

That is exactly what stigmergy predicts:

- early in the episode, no route exists yet
- later in the episode, the swarm reuses environmental information

Board takeaway: **The environment becomes part of the swarm’s shared memory.**

---

## Robustness and Limitations

### Robustness

Harder settings reduced performance, but did not destroy it.

Examples:

- more obstacles: **0.65 deliveries**
- 2 failed agents: **0.70**
- sensor noise: **1.05**
- control: **1.25**

### Limitations

- strongest scaling test used `1`, `3`, `6` agents
- robustness is partial, not universal
- strongest stigmergy result comes from repeated-source route reuse
- more physical validation would strengthen the project

Board takeaway: **The project is strong, but the limitations are real and clearly stated.**

---

## Real-World Impact

This type of swarm is promising for:

- disaster response
- weak-communication environments
- dynamic exploration tasks

One-line takeaway:

> **Swarms can operate where centralized systems fail.**

---

## Physical AI Demo

Show visually:

- physical robots
- mission-control interface
- digital pheromone visualization
- simulation screenshot

Minimal text:

> **Simulation → Real robots**

Presentation points:

- each robot acts locally
- there is no central planner telling each robot what to do
- stigmergic coordination is visible in real time

---

## Final Conclusion

> **Collective intelligence in this system emerges from decentralized interaction, and stigmergic communication is a key mechanism that makes that intelligence scale.**

In plain language:

- more robots alone do not explain the result
- better coordination explains the result
- stigmergy is one mechanism that enables that coordination

---

## Judge Summary

**Biggest result:**

My most important experiment tested whether the swarm becomes more efficient as it grows, and whether that improvement depends on stigmergic communication. The data show that larger swarms improve much more strongly when pheromone communication is enabled, especially after useful routes have already been discovered. That is evidence that the intelligence is emerging at the swarm level, not just from individual robots acting alone.

---

## Recommended Board Layout

### Left Column

- Title
- Motivation
- Research Question
- Hypothesis
- Why This Is Computer Science

### Center Column

- Bellman equation
- code snippet
- system diagram
- curriculum learning
- experimental design

### Right Column

- key result graph
- 2 supporting graphs
- impact
- demo
- conclusion

---

## Final Design Rule

**1 formula + 1 diagram + 2–3 graphs + 1 small code snippet**

If a section does not support one of these, cut it.

