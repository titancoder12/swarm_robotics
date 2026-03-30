# Trail Formation, RL, and Stigmergy

This note captures the design discussion around the intended swarm behavior:

- agents search for a target
- once an agent reaches the target and picks it up, it returns to the nest
- on the return path, it deposits pheromone
- repeated successful returns gradually form a trail
- later agents can use that trail to navigate to the target more efficiently

## 1. Intended Behavior

The desired policy is not only:

- find the target

It is the full loop:

1. explore until a target is found
2. pick up or otherwise interact with the target
3. return to the nest
4. deposit pheromone along the return route
5. allow later agents to exploit the resulting trail

Over time, this should create a route from nest to target that is easier for the swarm to reuse.

## 2. Why This Is Useful

This behavior is especially meaningful in scenarios such as disaster response or search-and-rescue in unknown terrain.

The core idea is:

- there is no reliable predefined map
- the first successful path may be expensive to discover
- once discovered, the path can be externalized into the environment
- later agents no longer need to rediscover the route from scratch

In that framing:

- target = victim, supply cache, exit point, or hazard site
- nest = safe zone, base station, triage point, or staging area
- pheromone trail = an emergent safe-route memory

## 3. Why This Is A Strong RL Objective

This behavior is aligned with reinforcement learning.

Early in training:

- target finding is highly uncertain
- the policy must explore
- random or exploratory behavior is necessary
- rewards are sparse and first discoveries are costly

Later in training:

- successful target-to-nest returns produce pheromone structure
- that structure makes the environment more informative
- later agents can follow the trail
- behavior shifts from exploration-heavy to exploitation-heavy

So the intended learning dynamic is:

- early phase: exploration
- later phase: exploitation

## 4. Why This Is More Than Standard Exploitation

This is still RL, but with a stigmergic mechanism layered on top.

In standard RL:

- exploitation usually means the policy has internally learned a good action preference

In this swarm setup:

- exploitation also happens through the environment
- the swarm writes route information into the world
- later agents read and exploit that shared external signal

So the pheromone trail acts like:

- shared environmental memory
- route hint
- decentralized coordination mechanism

## 5. Why This Is Scientifically Interesting

The important scientific story is not just:

- the agents learned to reach the target

It is:

- the swarm learned to externalize useful route knowledge
- the environment became more informative after successful returns
- later agents benefited from earlier agents’ success

That is what makes the behavior ant-like and stigmergic.

## 6. Critical Conditions For This To Work

For the system to actually learn the intended behavior, a few conditions matter:

1. Returning to the nest must matter
   - reward for delivery/return should be strong enough
   - pickup alone is not enough if the goal is route formation

2. Pheromone placement should be tied to meaningful behavior
   - ideally while carrying the target or during return-to-nest behavior
   - this avoids meaningless pheromone spam during aimless exploration

3. Pheromone following should help but not dominate
   - if it is too weak, trails will not matter
   - if it is too strong, agents may overfit to old trails and stop exploring

4. The curriculum should support the full loop
   - first teach target seeking
   - then target return
   - only later rely on multi-agent trail exploitation

5. Some exploration pressure should remain
   - otherwise the swarm may over-commit to an early, suboptimal route
   - this matters especially if the environment changes or if there are multiple possible routes

## 7. Interpretation In RL Terms

This design can be summarized as:

- exploration is required to discover targets
- successful return paths create pheromone traces
- pheromone converts past success into shared environmental information
- later policies exploit that information

So the full system combines:

- RL-based policy learning
- decentralized execution
- external shared memory via pheromone
- emergent route formation through stigmergy

## 8. Bottom Line

Yes, this behavior is conceptually aligned with RL.

More specifically:

- target search is the exploration-heavy part
- trail following is the exploitation-heavy part
- pheromone is the mechanism that allows the swarm to transition from expensive discovery to efficient reuse

This is a strong fit for the project because it supports:

- decentralized coordination
- unknown environments
- no required predefined map
- emergent route formation that becomes more useful over time

## 9. How To Achieve This In Training

To make this behavior emerge in practice, the training setup needs to teach the
full loop in the right order rather than hoping it appears all at once.

The most important idea is:

- do not train only “reach target”
- train “find target -> pick up -> return -> deposit -> later follow”

### 9.1 Make The Task Reward Match The Full Loop

The reward structure should emphasize the behavior you actually want.

That usually means:

1. target discovery / local detection
   - small shaping signal
   - enough to encourage search
   - not so large that touching a target becomes the whole task

2. pickup
   - meaningful reward
   - strong enough to teach that the target matters
   - but still smaller than successful return to nest

3. nest delivery / successful return
   - strongest core task reward
   - this is what teaches the agent that the route is not finished at pickup

4. pheromone following
   - small supportive reward only
   - should help exploit useful trails without replacing the actual task

5. pheromone deposit
   - either neutral or slightly costly per deposit
   - enough to discourage meaningless spam
   - but not so costly that agents avoid depositing on successful return paths

The key principle is:

- pickup teaches “this object matters”
- delivery teaches “the full route matters”

If pickup is rewarded strongly but delivery is weak, the system will tend to
learn approach behavior without stable return-route behavior.

### 9.2 Tie Pheromone To Meaningful Behavior

Pheromone should reinforce successful return paths, not random wandering.

The cleanest version is:

- agents deposit pheromone on the return trip to the nest
- especially while carrying the target

That makes the trail represent:

- a route that actually led to success

instead of:

- arbitrary motion history

If pheromone is deposited too freely during exploration, the environment fills
with noisy traces and the trail signal becomes much less useful.

### 9.3 Use Curriculum To Teach The Loop In Stages

The easiest way to achieve the final stigmergic behavior is to teach pieces of
the task progressively.

The intended curriculum logic is:

1. very easy single-agent stage
   - tiny world
   - one target
   - almost no obstacle complexity
   - teaches target seeking and pickup

2. harder single-agent stage
   - larger world
   - some obstacles
   - teaches search plus route finding around structure

3. small-swarm stages
   - medium and then larger environments
   - teaches scaling and interference handling

4. full-swarm acclimation stage
   - lets the larger swarm learn to operate in the larger world before final difficulty

5. full-swarm final stage
   - very large world
   - many obstacles
   - enough complexity that random wandering is no longer effective

This matters because the final desired behavior depends on several competencies:

- finding the target
- knowing to return
- depositing useful pheromone
- exploiting the trail collectively

Trying to learn all of that directly in the hardest environment is much less
likely to work.

### 9.4 Preserve Some Exploration Even After Trails Exist

Even after pheromone begins to work, training should not collapse into pure
trail-following.

Some continued exploration pressure is still useful because:

- the first discovered route may be suboptimal
- the environment may change
- the swarm may need to discover alternate routes
- multiple targets may appear in different places

So the learning dynamic should become:

- more exploitation over time
- but not zero exploration

In practice, this means the system should keep:

- some residual policy stochasticity during training
- only modest pheromone-follow reward

### 9.5 Measure The Right Outcome

To prove the trail is actually helping, evaluate more than just total reward.

Useful metrics include:

1. time to first target discovery
2. time from first discovery to later discoveries
3. pickup count versus delivery count
4. route efficiency on return to nest
5. pheromone usage over time
6. whether later agents reach the target faster after a trail exists

The strongest claim is not:

- “the agents got reward”

It is:

- “the first successful route created shared environmental information”
- “later agents used that information to improve navigation”

### 9.6 Training Recipe Summary

A practical recipe for this project is:

1. teach single-agent target finding in a tiny easy environment
2. teach single-agent return behavior in somewhat larger obstructed environments
3. make delivery more important than pickup
4. gate pheromone deposition to successful return-style behavior
5. keep pheromone-follow reward small but useful
6. scale to small swarm, then full swarm
7. evaluate whether later agents benefit from earlier successful routes

## 10. Bottom-Line Training Interpretation

In training terms, the intended system should work like this:

- early policy behavior is mostly exploration
- successful rollouts that find and return from the target create pheromone traces
- those traces make later trajectories easier
- later policies exploit the shared traces more effectively
- the swarm gradually shifts from expensive random search toward structured route reuse

That is the core mechanism by which the desired trail behavior can emerge.

## 11. Current System Alignment

The current implementation direction now matches this trail discussion more
closely than before:

- delivery reward is stronger than pickup reward by default
- carrying-food progress back toward the nest has its own small shaping term
- pheromone deposition is gated by carrying-food status by default
- pheromone deposition can also require actual progress toward the nest
- recurrent MAPPO curriculum remains the main recommended trail-learning path

That does not guarantee strong trail formation from a short run, but it means
the code is now aligned with the intended `discover -> return -> deposit ->
exploit` story rather than fighting it.
