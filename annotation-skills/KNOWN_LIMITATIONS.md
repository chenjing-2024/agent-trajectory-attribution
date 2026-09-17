# Known limitations

This first public release packages research annotation workflows. Machine `ok`
means the configured checks passed; it is not independently established ground
truth. The issues below remain unresolved in this release.

## Observed issues

- Canary unsafe: a live sample performed an injected action, yet attack_success
  and downstream judgements were inconsistent and automatic checks allowed it.
- Canary refusal: one live sample attributed the refusal to the benign user task
  instead of the component introducing the malicious instruction.
- AgentDojo unsafe: a repair replay moved C4 from execution_chain to attack_chain;
  semantic validation passed despite the questionable distinction. Target and
  primary attribution agreed. Its API compatibility fix does not settle this
  annotation-quality issue.
- Agent3Sigma unsafe: metadata may retain errors from before a successful repair.
  Use final validation reports and case outcomes when assessing final status.

## Coverage and operating requirements

- Five model-backed workflows have had small live API smoke tests using
  gpt-4.1-mini. This is not a benchmark accuracy study or broad provider test.
- AgentDojo task alignment is deterministic. Automatic constraint extraction
  supports limited email templates; other tasks need reviewed constraint files.
  Unknown constraints or unmet/unverified prerequisites require review.
- Compatible prepared inputs preserve component IDs and skip repeated conversion.
  Different source schemas can still require explicit dataset adapters.
- Dry runs check planning only. Offline tests use synthetic fixtures and mocked
  model stages; neither establishes real-model annotation quality.
- API-backed operation needs credentials and may incur provider charges, but
  does not require a local GPU. Local model serving has separate hardware needs.
- AgentDojo unsafe includes explicit capability profiles for documented model
  combinations. Other model/provider pairs require reviewed configuration; the
  adapter does not infer support from a model name prefix.
- Python 3.10+ and Bash are required as documented. Local verification used
  macOS and Python 3.13.2; Windows and other Python versions have not been verified.
- Some test fixtures use interpreter shebangs that do not support spaces in the
  Python executable path. Use a Python/virtualenv location without spaces for
  the documented test commands. Application paths with spaces were exercised.

## Distribution status

Only code, Skill instructions, synthetic examples and regression fixtures are
included. Real benchmark trajectories, API keys, local environment configuration
and live model outputs are not bundled. Original code and accompanying original
Skill documentation are licensed under [Apache-2.0](LICENSE-CODE). Benchmark
data and third-party material retain their separate licenses and notices.
