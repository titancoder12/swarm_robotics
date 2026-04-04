# [Step Title]
Work in an existing [project / codebase type] for [problem domain].

Task:
[One-sentence summary of the requested implementation or analysis.]

Main goals:
1. [Primary goal]
2. [Secondary goal]
3. [Tertiary goal]

Goals:
- [Desired outcome]
- [Desired outcome]
- [Desired outcome]
- Keep the repo runnable and changes minimally invasive.

--------------------------------
CONTEXT / BACKGROUND
--------------------------------

Current repo context:
- [Relevant subsystem or file]
- [Relevant subsystem or file]
- [Relevant subsystem or file]

Important current assumptions:
- [Existing action space / observation contract / runtime contract]
- [Existing training or deployment path]
- [Known compatibility constraint]

--------------------------------
IMPORTANT CONSTRAINTS
--------------------------------

- Modify the existing code in place.
- Reuse existing files and classes when possible.
- Keep naming/style consistent with the repo.
- Do not rewrite unrelated systems.
- Do not introduce breaking changes to [observation shapes / action spaces / deployment interfaces] unless explicitly required.
- Do not give pseudocode only; implement real code.

Optional domain-specific constraints:
- [Sim-to-real constraint]
- [No privileged information constraint]
- [Preserve backward compatibility where reasonable]

--------------------------------
PART 1 — [First Workstream]
--------------------------------

Problem / Goal:
- [What is wrong or missing]
- [Why it matters]

Required changes:
- [Required change]
- [Required change]
- [Required change]

Preferred implementation:
- In [file or subsystem]:
  - [Implementation guidance]
  - [Implementation guidance]
- Add config if needed:
  - `[config_field]: [type] = [default]`

Important:
- [Boundary / non-goal]
- [Compatibility requirement]

--------------------------------
PART 2 — [Second Workstream]
--------------------------------

Problem / Goal:
- [What is wrong or missing]
- [Why it matters]

Required changes:
- [Required change]
- [Required change]

Preferred implementation:
- In [file or subsystem]:
  - [Implementation guidance]
  - [Implementation guidance]

Also check:
- [Related file]
- [Related file]
- [Related file]

--------------------------------
PART 3 — [Metrics / Validation / Docs]
--------------------------------

Add or update:

1. Validation / correctness
- [Validation requirement]
- [Validation requirement]

2. Logging / metrics
- [Metric or log requirement]
- [Metric or log requirement]

3. Documentation / compatibility
- [Doc requirement]
- [Backward compatibility requirement]

--------------------------------
PART 4 — Implementation Style
--------------------------------

- Keep changes minimal and local.
- Preserve the existing architecture unless the task explicitly requires otherwise.
- Prefer config-driven parameters over hardcoded constants.
- Add concise comments only where the logic is non-obvious.
- Keep the code easy to inspect and debug.

--------------------------------
IMPLEMENTATION GUIDANCE
--------------------------------

Search for:
- [Relevant function / subsystem]
- [Relevant config or constants]
- [Relevant logging / plotting path]
- [Relevant inference / evaluation / runtime path]

Suggested approach:
- [Step 1]
- [Step 2]
- [Step 3]

Safety / quality constraints:
- [Avoid reward hacking / regressions / silent contract drift]
- [Keep runtime compatibility]
- [Keep output volume or dependencies under control]

--------------------------------
DELIVERABLES
--------------------------------

After implementation, provide:

1. List of modified files
2. Exact fields / APIs / behaviors added or changed
3. Short explanation of the new logic in plain English
4. Example commands to run or verify the change
5. Any caveats, compatibility notes, or retraining implications

--------------------------------
EXAMPLE EXPECTED EFFECT
--------------------------------

After the changes:
- [Observable outcome]
- [Observable outcome]
- [Observable outcome]

--------------------------------
IMPORTANT
--------------------------------

Keep the system runnable.
Implement real code, not pseudocode.
Do not introduce unrelated behavior changes.
