Work in the existing multi-agent swarm reinforcement learning codebase.

Task:
Fix checkpoint saving so training no longer creates a duplicate top-level `shared.pt` in `checkpoints/`, and make the user-provided name drive a single enclosing checkpoint folder.

Goal:
I want checkpoint outputs to be grouped cleanly under one named folder instead of spilling files directly into `checkpoints/`. The current duplicate `shared.pt` creation at the top level should be removed and prevented in code.

--------------------------------------------------
PART 1 — REMOVE DUPLICATE TOP-LEVEL `shared.pt`
--------------------------------------------------

Problem:
- Training is creating an extra top-level `shared.pt` in `checkpoints/`.
- This is confusing because the real milestone checkpoints already exist in structured subfolders.

Required fix:
- Stop writing duplicate checkpoint files directly into the bare `checkpoints/` root.
- Ensure the code does not recreate this duplicate behavior in future runs.
- Keep checkpoint saving working for shared-policy training.

Important:
- Do not break the existing milestone save behavior.
- Do not remove needed metadata; only fix the layout and duplicate save path logic.

--------------------------------------------------
PART 2 — RENAME THE CLI CONCEPT FROM `filename` TO `folder-name`
--------------------------------------------------

Change the user-facing training argument so the current naming input becomes:

- `--folder-name`

instead of:

- `--filename`

Requirements:
- `--folder-name` should be the main user-facing argument for checkpoint grouping.
- If practical, keep backward compatibility for `--filename` for now, but make `--folder-name` the preferred name.
- Update help text and docs so the intent is clear:
  - this value controls the enclosing checkpoint folder name
  - not just a generic output filename

--------------------------------------------------
PART 3 — REQUIRED CHECKPOINT LAYOUT
--------------------------------------------------

I want the checkpoint output structure to look like this:

```text
checkpoints/
  <folder_name>/
    metadata.json
    1_4_trained/
      shared.pt
      metadata.json
    1_2_trained/
      shared.pt
      metadata.json
    3_4_trained/
      shared.pt
      metadata.json
    full_policy/
      shared.pt
      metadata.json
```

For independent policies, keep the equivalent per-agent files inside those milestone folders.

Important:
- Everything that currently goes to milestone folders should remain in milestone folders.
- The enclosing folder should be named from `--folder-name`.
- Do not also create a duplicate top-level `shared.pt` directly under `checkpoints/`.

--------------------------------------------------
PART 4 — SAVE-DIR BEHAVIOR
--------------------------------------------------

Keep the current repo style, but make the layout deterministic:

- `--save-dir` should remain the base checkpoint root, for example:
  - `checkpoints`
- `--folder-name` should create a subfolder inside that base root

Example:

```bash
./.venv/bin/python train/independent_dqn_pytorch.py \
  --folder-name with_pheremone \
  --save-dir checkpoints
```

Should save into:

```text
checkpoints/with_pheremone/
```

and all milestone folders should live inside that directory.

--------------------------------------------------
PART 5 — IMPLEMENTATION GUIDANCE
--------------------------------------------------

Search for:
- training checkpoint save logic
- milestone save logic
- metadata writing
- `filename` argument handling
- `save_dir` handling

Likely files:
- `train/independent_dqn_pytorch.py`
- `train/experiment_utils.py`
- any docs or steps that reference training commands

Preferred implementation:
- Add a small helper that resolves the final checkpoint root:
  - `final_checkpoint_dir = <save_dir>/<folder_name>`
- Use that resolved folder consistently for:
  - milestone checkpoints
  - checkpoint metadata
- Make sure the final folder name is sanitized for filesystem safety.

--------------------------------------------------
PART 6 — QUALITY REQUIREMENTS
--------------------------------------------------

- Real implementation only, not pseudocode
- Keep the repo runnable
- Do not break milestone checkpoint saving
- Do not break shared vs independent policy saving
- Avoid unrelated refactors
- Prefer backward compatibility where practical

--------------------------------------------------
DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. List of modified files
2. Exact new checkpoint directory structure
3. Explanation of how duplicate top-level `shared.pt` creation was removed
4. Explanation of how `--folder-name` now works
5. Example command using `--folder-name`
6. Any backward-compatibility notes for `--filename`

--------------------------------------------------
IMPORTANT
--------------------------------------------------

Implement the change directly in the existing repo.
Keep it runnable.
Prevent the duplicate checkpoint file behavior from happening again in code.
