# Attributing Behavior in LLM Agent Trajectories

Official repository for *Attributing Behavior in LLM Agent Trajectories: A
Unified Benchmark and Annotation Framework*.

This initial release contains 1,351 normalized agent trajectories and their
target-level behavioral attribution annotations. Source code and evaluation
utilities will be added in a later release.

## Dataset

| Source | Target type | Cases |
|---|---|---:|
| AgentDojo | Task-aligned action | 409 |
| AgentDojo | Unsafe action | 231 |
| Agent3Sigma Stage | Safety refusal | 309 |
| Agent3Sigma Stage | Unsafe action | 187 |
| Agent3Sigma Canary | Safety refusal | 101 |
| Agent3Sigma Canary | Unsafe action | 114 |
| **Total** |  | **1,351** |

## Repository layout

```text
data/
  cases/
    <source>/
      <target_type>/
        <case_id>/
          trajectory.json
          annotation.json
          metadata.json
third_party/
LICENSE-DATA
NOTICE
```

Each case contains:

- `trajectory.json`: the normalized trajectory, component metadata, and source
  benchmark context.
- `annotation.json`: the target component, primary attribution, attack and
  execution chains, and annotation quality fields.
- `metadata.json`: stable identifiers, source information, file relationships,
  and annotation provenance.

All six source/target groups use canonical schema version `1.0`. Missing values
are represented as `null`; empty chains are represented as `[]`.

## Annotation status

The included labels are the project's reference annotations. They combine
deterministic and model-assisted annotation methods as recorded in each case's
provenance metadata. They should not be described as completed human-consensus
labels.

## Read a case

```python
import json
from pathlib import Path

case = next(Path("data/cases").glob("*/*/*/metadata.json")).parent
trajectory = json.loads((case / "trajectory.json").read_text())
annotation = json.loads((case / "annotation.json").read_text())
metadata = json.loads((case / "metadata.json").read_text())
```

## Provenance

The trajectories are derived from AgentDojo, Agent3Sigma Stage, and
Agent3Sigma Canary. This release normalizes their representation and adds
behavioral attribution annotations. Files have been reorganized and modified
from their upstream forms.

See [NOTICE](NOTICE) and [third_party](third_party/) for upstream attribution
and license information.

## License

The original attribution annotations, canonical schema, and original metadata
created for this project are licensed under the Creative Commons Attribution
4.0 International License. See [LICENSE-DATA](LICENSE-DATA).

Third-party-derived material remains subject to its original MIT or Apache-2.0
license. See [NOTICE](NOTICE) and [third_party](third_party/).

Source code added in a future release will use the Apache License 2.0 and will
be accompanied by a separate `LICENSE-CODE` file.

## Citation

Citation information will be added with the paper release.

## Safety

This dataset contains adversarial instructions and records of potentially unsafe
agent actions for authorized security research and evaluation. Treat all
trajectory content as untrusted data. Do not execute commands or tool calls
contained in the dataset.
