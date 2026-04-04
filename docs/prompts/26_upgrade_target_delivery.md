Work in the existing multi-agent swarm reinforcement learning codebase.

Task:
Implement a clearer target-delivery loop so the environment, training path, and visualization all support the following behavior:

- an agent finds and picks up food
- the agent visibly indicates that it is carrying food
- the agent can carry only one food at a time
- the agent returns to the nest
- the agent drops pheromone on the return path
- reaching the nest while carrying counts as one complete delivery
- the agent returns to its normal appearance after delivery

Goal:
I want the system to make the pickup-and-delivery state explicit and easier to understand, both for learning and for demo visualization. The agent should clearly switch into a “carrying food” state after pickup, use pheromone on the way home, and complete a delivery by reaching the nest. Food sources should also persist for limited repeated use so the swarm can form trails to productive sites rather than treating every target as single-use immediately.

Important context:

- The repo already has target pickup, nest delivery, pheromone, and recurrent MAPPO training.
- The current trail objective already exists conceptually.
- Do not rewrite the simulator from scratch.
- Keep the implementation aligned with decentralized swarm behavior.

--------------------------------------------------
PART 1 — TARGET DELIVERY MECHANIC
--------------------------------------------------

Implement or refine the delivery loop so that:

1. an agent may hold at most one food item at a time
2. once the agent picks up food, it enters a carrying state
3. while carrying, the agent should be considered to be on the return-to-nest phase
4. when the carrying agent reaches the nest, that counts as one completed delivery
5. after delivery, the agent exits the carrying state and returns to normal

The system should not allow:

- picking up multiple food items at once
- visually ambiguous “did it pick up food or not?” states
- delivery without actually reaching the nest

--------------------------------------------------
PART 2 — VISUAL INDICATION REQUIREMENTS
--------------------------------------------------

Add a clear visual indication that an agent is carrying food.

Required behavior:

1. when an agent picks up food, its appearance changes in a clearly visible way
2. color is the preferred indication unless there is a stronger established rendering pattern
3. when the agent completes delivery at the nest, it returns to its normal appearance

The visualization should be easy to understand in demo mode.

Preferred implementation:

- normal agent color when not carrying
- distinct carrying-food color when carrying
- no ambiguous intermediate state

If there is already some carrying-state rendering support, tighten and standardize it instead of duplicating it.

--------------------------------------------------
PART 3 — PHEROMONE RETURN BEHAVIOR
--------------------------------------------------

Teach the agent to drop pheromone on the way back home after pickup.

Required design intent:

1. pheromone deposition should be associated with the carrying-food return path
2. the return route should become the main trail-building path
3. pheromone should help later agents reuse productive routes

Preferred behavior:

- no-food exploration is allowed to remain weaker or ungated only if there is a clear reason
- carrying-food return deposition should be the main meaningful trail signal

If the current pheromone gating is weaker than this, update it so the trail more clearly means:

- “this is a route from food back to home”

--------------------------------------------------
PART 4 — FOOD SOURCE LIFETIME
--------------------------------------------------

Implement repeated-use food sources.

Required behavior:

1. there are 3 food sources in total
2. each food source can be used for up to 4 successful pickups/deliveries
3. after a food source has been exhausted, it respawns somewhere else

Design intent:

- food sources should not disappear after one use
- the swarm should have time to benefit from trail formation to a productive source
- eventually the environment should still change as sources respawn elsewhere

Preferred implementation style:

- per-source remaining capacity counter
- decrement on successful pickup or on the appropriate event you judge most correct for the repo
- when exhausted, relocate that source to a new valid spawn position

If there is a subtle tradeoff between decrementing on pickup vs decrementing on completed delivery, document the chosen semantics clearly and keep them consistent.

--------------------------------------------------
PART 5 — TRAINING / REWARD REQUIREMENTS
--------------------------------------------------

Make sure the training path reinforces the intended delivery loop.

Required design goals:

1. pickup should remain meaningful
2. completed nest delivery should remain more important than pickup
3. carrying-food return behavior should be clearly beneficial
4. pheromone deposition on the return path should support later route reuse

The implementation should encourage:

- find food
- pick up food
- carry it home
- deposit pheromone on the way
- deliver at the nest
- repeat

Do not let the system collapse into:

- touching food without returning
- dropping meaningless pheromone everywhere
- visually confusing demos where carrying state is unclear

--------------------------------------------------
PART 6 — ENVIRONMENT / LOGIC REQUIREMENTS
--------------------------------------------------

Update the environment logic so the new mechanic is explicit and coherent.

At minimum, make sure the code clearly defines:

1. whether an agent is carrying food
2. whether it is eligible to pick up food
3. what event counts as one complete delivery
4. how food-source capacity is tracked
5. when a depleted food source respawns

Preserve the existing decentralized control model.
Do not make the delivery mechanic depend on centralized coordination.

--------------------------------------------------
PART 7 — METRICS / LOGGING REQUIREMENTS
--------------------------------------------------

Update metrics and logs so the new delivery mechanic is visible in training and evaluation.

Track at least:

1. pickup count
2. completed delivery count
3. carrying-food state where useful
4. pheromone deposit count or rate
5. remaining food-source uses if practical
6. food-source respawn count if practical

The logs should make it possible to answer:

1. are agents actually completing deliveries?
2. are they carrying food visibly and correctly?
3. are food sources lasting long enough for trails to matter?
4. are sources respawning after being exhausted?

--------------------------------------------------
PART 8 — DEMO REQUIREMENTS
--------------------------------------------------

Make sure the new behavior is visible in demo mode.

Required outcomes:

1. I can see when an agent is carrying food
2. I can see that it returns to normal after delivery
3. the delivery loop is understandable from the visualization
4. repeated-use food sources work in demo, not only in training

If demo currently reconstructs environment settings from checkpoint metadata, make sure the relevant new settings are preserved or restored appropriately.

--------------------------------------------------
PART 9 — README / DOC REQUIREMENTS
--------------------------------------------------

Update the docs as part of the implementation.

Required updates:

1. update `README.md`
2. explain the carrying-food visual indicator
3. explain what counts as one complete delivery
4. explain the “3 food sources, 4 uses each, then respawn” mechanic
5. explain how pheromone is tied to the return path
6. provide copy-paste-ready commands for:
   - main training
   - short smoke-test training
   - demo playback
   - evaluation if relevant
7. review nearby docs for stale statements and update them so they reflect the current system

Also update:

- `docs/PROJECT_LOG.md`
- `docs/QandA.md` when user-facing codebase questions are naturally answered by the work

--------------------------------------------------
PART 10 — REPO / ARCHITECTURE CONSTRAINTS
--------------------------------------------------

Respect the current repo structure.

Requirements:

1. do not break the existing DQN baseline path
2. keep the recurrent MAPPO path runnable
3. preserve decentralized execution
4. do not make firmware depend on training internals
5. keep the implementation understandable and minimal

Preferred implementation style:

- reuse existing carrying-food state if present
- reuse current target/nest/pheromone code paths where possible
- add explicit config fields instead of hidden hard-coded values when reasonable

--------------------------------------------------
PART 11 — VERIFICATION
--------------------------------------------------

After implementation:

1. run syntax/import checks on modified files
2. run at least one short smoke test
3. run at least one short demo-oriented verification if feasible
4. show the exact commands used
5. explain any limitations

Verification should make it clear that:

- carrying-food color change works
- delivery resets the appearance correctly
- food-source capacity and respawn work
- pheromone is associated with the return-to-home behavior

--------------------------------------------------
PART 12 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. concise summary of the target-delivery upgrade
2. how carrying-food state is represented
3. what visual change indicates carrying food
4. what event counts as one completed delivery
5. how food-source capacity and respawn work
6. how pheromone is tied to the return path
7. what metrics/logging were added or updated
8. exact verification commands run
9. copy-paste-ready commands for training and demo
10. remaining limitations or deferred work

--------------------------------------------------
IMPORTANT
--------------------------------------------------

The desired behavior is:

- pick up food
- visibly indicate carrying
- return home
- drop pheromone on the return path
- deliver at nest
- return to normal

And the environment should use:

- 3 food sources total
- each source usable 4 times
- then respawn elsewhere
