# Prompt 47: Add Non-Carrying Exploration Randomness

Implement a stage-configurable mechanism that adds extra decision randomness to agents only while they are exploring, meaning they are not carrying food back to the nest.

Requirements:
- Add configuration support for a non-carrying exploration random-action probability.
- Apply the randomness env-side so it affects both training and demo consistently.
- Only apply it to agents that are not carrying food.
- Prefer movement-producing exploration actions rather than stop-like or deposit actions.
- Keep carrying-food return-to-nest behavior untouched.
- Preserve the existing force-explore nest-exit behavior; if both features apply, the explicit nest-exit override should still win.
- Enable this new randomness in the later swarm curriculum stages where empty-agent exploration matters most.
- Log metrics showing how often the randomness was active and how often it overrode the requested action.
- Save and restore the setting through MAPPO checkpoint metadata so demo reproduces it.

Also update the relevant docs to reflect the new exploration-randomness behavior and add a short project-log entry.
