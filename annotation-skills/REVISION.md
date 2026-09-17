# First public release

This release provides three independently installable annotation Skills,
synthetic examples, and offline regression tests. The release preparation changes
only licensing and documentation; annotation scripts, prompts, quality gates,
dependencies, and tests are unchanged.

- Include a Chinese overview, known limitations, and a synthetic quick start.
- Keep downstream attribution tools outside this annotation collection.
- Exclude caches, private configuration, run outputs, and real benchmark data.
- License original code and accompanying original Skill documentation under
  [Apache-2.0](LICENSE-CODE); preserve separate data and third-party licensing.

Validation coverage includes 81 offline tests, all three Skill structure checks,
and small live API smoke tests using GPT-4.1-mini. The AgentDojo unsafe full-run
retest passed. A separate repair replay called the API successfully, but its C4
chain classification remains under review. See [known limitations](KNOWN_LIMITATIONS.md).
