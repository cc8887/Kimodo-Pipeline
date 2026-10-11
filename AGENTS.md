# Production workspace

- This repository owns the Kimodo production tools, rules, Viewer and tutorials. The Godot project remains the consumer and owns its runtime, import adapters, demo and scene tests.
- Read `README.md`, `docs/入门教程.md` and `docs/architecture/standalone-workspace.md` before changing entrypoints or data boundaries.
- Preserve existing changes. Stage explicit file lists. Commit and push require their respective user authorization.
- Original ALS reference exports, models, tokens, generated resources, caches and runtime receipts stay outside Git. Preserve JSON/FBX/raw byte hashes; do not format reference exports.
- A generation request runs the authoring DAG, independent successful branch conversion, Root/Pelvis processing, curves, final FBX readback, Godot import and related runtime checks by default. Only explicit diagnostics may stop at authoring.
- Keep original quality gates and negative controls. Source failure, missing assets, unsupported checks and failed game scenes remain visible; a diagnostic pack does not become a production release.
- Never overwrite an existing production version or runtime evidence directory. Viewer batch provenance must not borrow missing stages from historical outputs.
- Validate consumer changes in the current Godot project, using Optimize builds and relevant real scenes. Distinguish implementation, controlled tests, original comparisons and manual acceptance.
- Record current evidence in `docs/verification/`; keep this file limited to enduring constraints.
