# Model request parameters

AgentDojo unsafe uses `scripts/unsafe/model_requests.py` for initial annotation,
semantic validation, annotation repair, and both JSON repair paths. Task alignment
remains deterministic and does not call a model.

On the official OpenAI endpoint, built-in profiles cover:

| Model | Reasoning effort | Output limit | Temperature |
| --- | --- | --- | --- |
| gpt-4.1, gpt-4.1-mini, gpt-4.1-nano and their 2025-04-14 snapshots | Omitted, including when the CLI requests `none` | max_tokens by default | Supported |
| gpt-5.4-mini | none, low, medium, high, xhigh | max_completion_tokens | Omitted |

Repair requests use JSON object response format. Semantic validation requests low
reasoning effort and repair defaults to medium; these settings apply only when
the selected profile supports reasoning. `--use_max_completion_tokens` changes
the semantic request's token-limit field only; it does not enable reasoning.
Token budgets still include reasoning tokens for reasoning models.

Other model/provider pairs require an explicit reviewed capability file. Set
`AGENTDOJO_MODEL_CAPABILITIES` to its path; child stages inherit the setting.
For example, adapt this profile to the actual server's documented support:

```json
{
  "models": {
    "your-local-model": {
      "base_url": "http://localhost:8000/v1",
      "token_parameters": ["max_tokens"],
      "temperature": true,
      "reasoning_efforts": [],
      "json_object": false
    }
  }
}
```

The endpoint must match exactly, excluding a trailing slash. Optional `extra_body`
is for explicitly configured provider extensions only. No local-server thinking
extension is automatically added; official OpenAI profiles reject extensions.
Unknown capabilities fail explicitly before a request. API errors remain errors:
the adapter does not drop parameters after an error or substitute another model.

Optionally set `AGENTDOJO_REQUEST_AUDIT` to a JSONL output path outside the skill.
It records stage, model, parameter names, non-sensitive parameter values, call
status, and token usage when returned. It excludes prompts, responses, credentials,
headers, endpoint URLs and provider-extension values. This is a request audit,
not an annotation quality result; use `quality_summary.json` for the latter.

Built-in profiles are based on the official model documentation:
- https://developers.openai.com/api/docs/guides/latest-model?model=gpt-4.1
- https://developers.openai.com/api/docs/models/gpt-5.4-mini

New profiles require verification against the selected model/provider's docs.
