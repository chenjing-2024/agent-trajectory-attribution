# Agent Trajectory Annotation Skills

[中文说明](README.zh-CN.md)

First public release of the annotation Skill collection, licensed under
[Apache-2.0](LICENSE-CODE). Read [known limitations](KNOWN_LIMITATIONS.md)
before interpreting machine validation as annotation correctness.
See [release notes](REVISION.md).

A collection of three independently installable skills for component-level
annotation of agent trajectories. Each skill packages workflow instructions,
schemas, and executable annotation and validation tools.

| Skill | Input / workflows | Entry point |
| --- | --- | --- |
| `agent3sigma-annotation` | Agent3Sigma unsafe successful attacks and safety refusals | [SKILL.md](skills/agent3sigma-annotation/SKILL.md) |
| `agentdojo-annotation` | AgentDojo unsafe trajectories and deterministic task alignment | [SKILL.md](skills/agentdojo-annotation/SKILL.md) |
| `canary-annotation` | Normalized Agent3Sigma-Canary trajectories, unsafe actions and safety refusals | [SKILL.md](skills/canary-annotation/SKILL.md) |

## Install

Use Python 3.10 or newer. Canary also requires Bash. From this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For Codex, copy the desired skill folder into your skills directory. Run the
following in Bash or Zsh to install all three; it stops before copying if any
destination already exists:

```bash
python - <<'PY'
import os
import shutil
from pathlib import Path

source = Path('skills')
destination = Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex') / 'skills'
skills = [source / name for name in (
    'agent3sigma-annotation', 'agentdojo-annotation', 'canary-annotation'
)]
for skill in skills:
    if not (skill / 'SKILL.md').is_file():
        raise SystemExit(f'Missing skill: {skill}; run from the repository root')
    if (destination / skill.name).exists():
        raise SystemExit(f'Already installed: {destination / skill.name}')
destination.mkdir(parents=True, exist_ok=True)
for skill in skills:
    shutil.copytree(skill, destination / skill.name)
    print(f'Installed {skill.name}')
PY
```

Each skill includes its own `requirements.txt` for selective installation.
Run the agent in an environment where these dependencies are available. Keep
the complete skill folder together: scripts and references use relative paths.

## Use

Example requests (replace paths with your actual input and a new output directory):

```text
Use $agent3sigma-annotation to annotate unsafe successful-attack trajectories
from /path/to/cases into /path/to/new-run. Start with a dry run.

Use $agentdojo-annotation to produce task-alignment annotations from
/path/to/trajectories into /path/to/new-run.

Use $canary-annotation with /path/to/canary.env to validate the unsafe
annotations and report cases requiring repair.
```

Input formats are documented in each skill's `references/schemas.md`.
Model-backed stages require a provider-supported model and credentials supplied
through the configured environment variable. No dataset or credentials are bundled.
Dry runs do not produce annotations or establish annotation quality.

For Canary, copy `skills/canary-annotation/config/annotation.example.env` to a
run directory outside the skill, then set:

```bash
export PROJECT_ROOT=/absolute/path/to/new-canary-run
export MODEL_ID=your-provider-model-id
export ANNOTATION_CONFIG=/absolute/path/to/canary.env
```

Edit the copied config to match your normalized inputs and provider URL. Supply
the API key through `OPENAI_API_KEY` (or the configured variable), then run:

```bash
bash skills/canary-annotation/bin/annotation_skill.sh check-config
```

`check-config` checks configuration, local paths, and credential presence; it
does not test the provider. Follow the [Canary workflow guide](skills/canary-annotation/references/workflows.md)
for annotation, repair, and final validation. Leave `EXPECTED_*` assertions empty
unless counts are independently known.

## Repository layout

```text
skills/
  agent3sigma-annotation/   # Independent skill
  agentdojo-annotation/    # Independent skill
  canary-annotation/       # Independent skill
tests/
  agent3sigma/
  agentdojo/
```

This distribution contains annotation tools only. Optional downstream attribution
tools remain separate from this Skill collection.

See [examples/README.md](examples/README.md) for a synthetic, no-API quick start.

## Local checks

All six annotation workflows have a dry-run entry point:

| Skill / workflow | Runner under the skill folder | Dry-run options |
| --- | --- | --- |
| Agent3Sigma unsafe action | `scripts/unsafe/run_full_pipeline.py` | `--dry_run` |
| Agent3Sigma safety refusal | `scripts/safety_refusal/run_pipeline.py` | `--dry_run` |
| AgentDojo unsafe | `scripts/unsafe/run_full_pipeline.py` | `--dry_run` |
| AgentDojo task alignment | `scripts/task_alignment/run_pipeline.py` | `--dry_run` |
| Canary unsafe action | `bin/annotation_skill.sh unsafe-all` | `--dry_run --plan /path/to/new-plan.json` |
| Canary safety refusal | `bin/annotation_skill.sh safety-refusal-all` | `--dry_run --plan /path/to/new-plan.json` |

Supply the input/output and model arguments documented in each workflow guide.
Dry runs need no GPU or API key and do not execute annotation stages. Use a
placeholder model name when only checking command planning. Canary configuration
still requires an existing selected input path. Its missing intermediate
artifacts are recorded as deferred, and its repair shell runner is a single
planned stage. AgentDojo's full unsafe runner likewise records the initial
pipeline as a top-level stage; its initial runner can be dry-run separately.

Run from the repository root:

```bash
python -m unittest discover -s tests/output -v
python -m unittest discover -s tests/agent3sigma -v
python -m unittest discover -s tests/agentdojo -v
python -m unittest discover -s tests/canary -v
python -m unittest discover -s tests/normalization -v
```

These cover pipeline dry runs and selected merge/input-validation behavior.
They do not run paid model calls or validate results on a real dataset.

For supported AgentDojo unsafe model/provider combinations, see
[model parameters](skills/agentdojo-annotation/references/model_parameters.md).

## Contributing

Keep workflow-specific guidance in the relevant skill and detailed schemas in
its references. Keep each skill independently usable. Run the local checks when
changing scripts, and describe any changes to input or output schemas.

Do not commit credentials, local configuration, raw trajectories, model responses,
or run outputs. Use small synthetic fixtures when adding regression tests.

## License

The original code, Skill instructions, prompt templates, and accompanying
original documentation in this collection are licensed under
[Apache License 2.0](LICENSE-CODE).

Benchmark data and annotations retain their separate `LICENSE-DATA` terms in
the parent repository. Third-party material retains its original licenses and
notices; this code license does not replace them.
