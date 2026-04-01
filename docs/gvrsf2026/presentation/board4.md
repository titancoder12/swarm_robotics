# GVRSF Board Draft Optimized (Fully Aligned Version)

---

# 🎯 Core Design Rule

**1 formula + 1 diagram + 2–3 graphs + 1 small code snippet**

If something does not support one of these → remove it.

---

## Title

**Stigmergy Is All You Need**

**Learning decentralized swarm intelligence through environmental communication**

**A multi-agent reinforcement learning swarm learns to explore, find targets, and return them to a nest without a central controller.**

---

## Motivation

**Problem:** Most robot systems rely on central control\
**Question:** Can intelligence emerge without it?\
**Idea:** Use stigmergy — agents coordinate through the environment

---

## 🐜 Origin Story (Hook)

> Observed carpenter ants exploring without coordination\
> → Asked: can intelligence emerge from simple local rules?

Use video during presentation (10–15s):

- simple individual behavior
- organized collective outcome

---

## Research Question

> **Does stigmergic communication enable a decentralized swarm to scale better with size?**

---

## Hypothesis

- Pheromone > no pheromone
- Gap increases with swarm size
- Strongest effect after discovery

---

## 💡 What Is New

- RL + stigmergy combined in one system
- curriculum learning enables full task completion
- controlled scaling experiment shows causal effect

---

# 🧠 Reinforcement Learning Signals

**Policy:** π(a | o)

**Key ideas:**

- discounted reward (γ)
- advantage (A)
- PPO clipping (stability)

**Architecture:**

- Actor–Critic
- PPO
- GRU (memory under partial observation)

---

# 🧠 Depth Anchor (Bellman)

**V(s) = E[r + γV(s')]**

Good actions improve future outcomes, not just immediate reward.

Why this matters:

- exploration pays off later
- pheromones help future agents
- delivery is delayed reward

👉 **This is why stigmergy works**

---

# ⚙️ System Overview

**Task loop:** explore → detect → pick up → return → deliver

**Diagram:** Observation → Policy → Action\
↓\
Environment (pheromone)\
↓\
Next Observation

(Add nest, food, pheromone in visual)

---

# 🧪 Experimental Design

**Independent variables:**

- swarm size
- pheromone on/off

**Dependent variables:**

- deliveries
- late deliveries

**Controls:**

- same seeds
- same environment

👉 **Controlled comparison: only pheromone differs**

---

# 📊 RESULTS

## ⭐ KEY RESULT — Stigmergy Enables Scaling

At **6 agents**:

- With pheromone: **1.32 deliveries**
- Without: **0.00 deliveries**

**p < 0.01**

👉 **Coordination emerges only with stigmergy**

---

## Supporting Result — Curriculum Learning

- 0.00 → **1.25 deliveries**

👉 Training strategy is critical

---

## Supporting Result — Baselines

- MAPPO: **1.25**
- Rule-based: **0.30**
- Random: **0.05**

👉 Behavior is learned, not scripted

---

# 🧠 Why This Works

Strongest effects appear in:

- late deliveries
- post-discovery behavior

👉 Environment acts as shared memory

---

# 💻 Code Signal

```
delta = r + gamma * V(next) - V(current)
```

Shows value learning and long-term reasoning

---

# 🌍 Impact

- disaster response
- weak communication environments
- dynamic exploration

> **Swarms can operate where centralized systems fail**

---

# 🤖 Demo

- physical robots
- pheromone visualization
- simulation

> Simulation → Real robots

---

# ⚠️ Limitations

- limited swarm sizes tested
- partial robustness
- strongest results in route-reuse task

---

# 🏆 Judge Summary

**Key result:**

Larger swarms improve much more strongly when stigmergic communication is enabled, especially after routes are discovered. This shows intelligence emerges at the swarm level, not just from individual agents.

---

# 🧩 Layout

**Left:**

- title
- motivation
- origin story
- research question

**Center:**

- RL signals
- Bellman
- diagram
- experiment

**Right:**

- key result (BIG)
- 2 supporting graphs
- impact
- demo

---

# 🎤 Presentation Strategy

- Point → formula → explain Bellman
- Point → graph → explain result
- Point → diagram → explain system

Let judges ask deeper questions

---

# 🏅 Rubric Alignment (GVRSF)

**Scientific Thought:**

- controlled variables
- causal experiment
- statistical significance

**Originality:**

- RL + stigmergy
- new experiment design

**Communication:**

- visual-first
- minimal text
- clear hierarchy

---

# 💡 Final Message

You are not showing everything.

You are guiding judges to: → understand quickly\
→ ask questions\
→ discover your depth

