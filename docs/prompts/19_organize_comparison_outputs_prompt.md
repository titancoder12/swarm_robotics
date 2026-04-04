Work in the existing multi-agent swarm reinforcement learning codebase.

Task:
Reorganize the evaluation-comparison output structure so each comparison condition gets its own directory under `experiments/experiment_data/`, with separate `exploration_graphs`, `graphs`, and `raw` outputs.

Goal:
I want the output files from the four comparison conditions to be easy to inspect independently, instead of mixing everything into one shared `raw/`, `graphs/`, and `exploration_graphs/` directory.

Use these exact condition directory names:

- `trained_pheremone_use_pheremone`
- `trained_pheremone_no_use_pheremone`
- `trained_without_pheremone`
- `random_walk`

--------------------------------------------------
PART 1 — DIRECTORY STRUCTURE
--------------------------------------------------

Update the comparison output logic so it writes under:

`experiments/experiment_data/`

with this structure:

```text
experiments/experiment_data/
  trained_pheremone_use_pheremone/
    raw/
    graphs/
      PNG/
      PDF/
    exploration_graphs/
      PNG/
      PDF/
  trained_pheremone_no_use_pheremone/
    raw/
    graphs/
      PNG/
      PDF/
    exploration_graphs/
      PNG/
      PDF/
  trained_without_pheremone/
    raw/
    graphs/
      PNG/
      PDF/
    exploration_graphs/
      PNG/
      PDF/
  random_walk/
    raw/
    graphs/
      PNG/
      PDF/
    exploration_graphs/
      PNG/
      PDF/
```

Requirements:
- Auto-create all directories if they do not exist.
- Use relative repo paths, not hardcoded absolute paths.
- Keep the existing top-level output root configurable via `--output-dir`.

--------------------------------------------------
PART 2 — CONDITION TO DIRECTORY MAPPING
--------------------------------------------------

Map the comparison conditions to directory names as follows:

1. `trained_with_pheromone__eval_with_pheromone`
   -> `trained_pheremone_use_pheremone`

2. `trained_with_pheromone__eval_without_pheromone`
   -> `trained_pheremone_no_use_pheremone`

3. `trained_without_pheromone__eval_without_pheromone`
   -> `trained_without_pheremone`

4. `random_walk`
   -> `random_walk`

Important:
- Use the exact directory spellings above, even though they contain the repo/user spelling `pheremone`.
- Keep the internal comparison labels unchanged in CSV contents and legends unless there is a strong repo-wide reason to rename them.

--------------------------------------------------
PART 3 — RAW OUTPUTS
--------------------------------------------------

For each condition directory, save:

- per-episode raw CSVs under:
  - `<condition_dir>/raw/`

At minimum, save one raw CSV for that condition, for example:

- `experiments/experiment_data/trained_pheremone_use_pheremone/raw/<filename>_raw.csv`
- `experiments/experiment_data/trained_pheremone_no_use_pheremone/raw/<filename>_raw.csv`
- `experiments/experiment_data/trained_without_pheremone/raw/<filename>_raw.csv`
- `experiments/experiment_data/random_walk/raw/<filename>_raw.csv`

If the repo currently also writes a master combined CSV, keep it if useful, but the per-condition raw CSVs in the condition directories are required.

--------------------------------------------------
PART 4 — COMPARISON GRAPHS
--------------------------------------------------

For each condition directory, save its graph outputs under:

- `<condition_dir>/graphs/PNG/`
- `<condition_dir>/graphs/PDF/`

Requirements:
- Save graph outputs in both PNG and PDF formats.
- Keep matplotlib only.
- Use readable filenames.

Important:
- If the repo currently creates combined multi-condition comparison plots, do not remove them unless necessary.
- But also save the relevant graph files into the condition-specific graph folders so each condition directory is self-contained.

--------------------------------------------------
PART 5 — EXPLORATION GRAPHS
--------------------------------------------------

For each condition directory, save exploration outputs under:

- `<condition_dir>/exploration_graphs/PNG/`
- `<condition_dir>/exploration_graphs/PDF/`

Requirements:
- Save exploration visuals in both PNG and PDF formats.
- Do not save only mixed top-level files.
- Use consistent filenames that include the agent count when relevant.

Example:
- `experiments/experiment_data/random_walk/exploration_graphs/PNG/<filename>_agents_10.png`
- `experiments/experiment_data/random_walk/exploration_graphs/PDF/<filename>_agents_10.pdf`

--------------------------------------------------
PART 6 — IMPLEMENTATION GUIDANCE
--------------------------------------------------

Search for:
- comparison output directory creation
- raw CSV writing
- graph saving
- exploration visual saving
- comparison condition labels

Likely files:
- `analysis/evaluate_comparison.py`
- any helper used for saving CSVs or plots

Implementation preferences:
- Add a small helper to map `comparison_label -> condition_output_dir_name`
- Add helpers for:
  - per-condition raw directory
  - per-condition graph PNG/PDF directories
  - per-condition exploration PNG/PDF directories
- Keep the existing evaluation logic intact
- Do not rewrite the experiment architecture

--------------------------------------------------
PART 7 — QUALITY REQUIREMENTS
--------------------------------------------------

- Real implementation only, not pseudocode
- Keep the repo runnable
- Do not break current CSV contents
- Do not break the existing comparison labels
- Continue saving PNG and PDF where already supported
- Prefer minimal, focused changes

--------------------------------------------------
DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. List of modified files
2. Exact directory structure now produced
3. Mapping from comparison labels to output directories
4. Description of what is saved in each of:
   - `raw/`
   - `graphs/PNG/`
   - `graphs/PDF/`
   - `exploration_graphs/PNG/`
   - `exploration_graphs/PDF/`
5. Example command to run the comparison and produce the organized outputs

--------------------------------------------------
IMPORTANT
--------------------------------------------------

Implement the change directly in the existing repo.
Keep it runnable.
Use the exact directory names requested above.
