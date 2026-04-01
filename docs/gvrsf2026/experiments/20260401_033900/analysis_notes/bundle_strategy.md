# Bundle Strategy

This timestamped folder is a **federated full bundle** built to serve as a self-contained experiment package.

Why this was the right execution strategy:

- The repository already contained a strong completed broad campaign in `20260401_014801`.
- The repository also contained a stronger later matched-training pheromone campaign in `20260401_023140`.
- A fresh broad rerun started during this execution but did not begin producing results in a reasonable window and would have duplicated the weaker parts of the evidence instead of improving the science.
- Reusing the strongest validated subcampaigns with explicit provenance produces a higher-quality final package than forcing a rushed all-in-one rerun.

The integrated report in this folder therefore treats:

- `broad/` as the broad experiment family source
- `killer_scaling/` as the dedicated stigmergy scaling source

The top-level `report.md` is the self-contained science-fair narrative that explains how these subcampaigns fit together.
