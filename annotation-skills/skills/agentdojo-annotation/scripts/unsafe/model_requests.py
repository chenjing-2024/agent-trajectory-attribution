"""Chat request capabilities shared by annotation, semantic review and repair.

Unknown model/provider pairs require AGENTDOJO_MODEL_CAPABILITIES (JSON file).
No API-error-driven parameter dropping or model substitution is performed.
"""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


class ModelCapabilityError(ValueError):
    pass


def capabilities(model, base_url):
    endpoint = str(base_url).rstrip('/')
    official = urlsplit(endpoint).hostname == 'api.openai.com'
    path = os.environ.get('AGENTDOJO_MODEL_CAPABILITIES')
    profile = None
    if path:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict) or not isinstance(payload.get('models'), dict):
            raise ModelCapabilityError('Capability file requires a models object')
        profile = payload['models'].get(model)
        if profile is not None and (not isinstance(profile, dict) or profile.get('base_url') != endpoint):
            raise ModelCapabilityError('Capability profile base_url must match the actual endpoint exactly')
    if profile is None and official:
        classic = {'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano'}
        classic |= {name + '-2025-04-14' for name in tuple(classic)}
        if model in classic:
            profile = {'token_parameters': ['max_tokens', 'max_completion_tokens'],
                       'temperature': True, 'reasoning_efforts': [], 'json_object': True}
        elif model == 'gpt-5.4-mini':
            profile = {'token_parameters': ['max_completion_tokens'], 'temperature': False,
                       'reasoning_efforts': ['none', 'low', 'medium', 'high', 'xhigh'], 'json_object': True}
    if profile is None:
        raise ModelCapabilityError('Unknown model/provider capabilities; set AGENTDOJO_MODEL_CAPABILITIES to a reviewed JSON profile')
    allowed = {'base_url', 'token_parameters', 'temperature', 'reasoning_efforts', 'json_object', 'extra_body'}
    if set(profile) - allowed:
        raise ModelCapabilityError('Unsupported capability fields')
    tokens = profile.get('token_parameters')
    efforts = profile.get('reasoning_efforts')
    if not isinstance(tokens, list) or not tokens or any(x not in ('max_tokens', 'max_completion_tokens') for x in tokens):
        raise ModelCapabilityError('token_parameters must list supported token limit names')
    if type(profile.get('temperature')) is not bool or type(profile.get('json_object')) is not bool:
        raise ModelCapabilityError('temperature and json_object capabilities must be booleans')
    if not isinstance(efforts, list) or any(not isinstance(x, str) or not x for x in efforts):
        raise ModelCapabilityError('reasoning_efforts must be a list of supported strings, or []')
    extra = profile.get('extra_body', {})
    if not isinstance(extra, dict):
        raise ModelCapabilityError('extra_body must be an object')
    if set(extra) & {'model', 'messages', 'temperature', 'reasoning_effort', 'max_tokens', 'max_completion_tokens', 'response_format', 'timeout', 'api_key'}:
        raise ModelCapabilityError('Provider extensions cannot override standard request fields or contain api_key')
    if official and extra:
        raise ModelCapabilityError('Provider-specific extra_body is not allowed on the official OpenAI endpoint')
    return profile


def request_kwargs(*, model, base_url, messages, temperature=None, output_tokens=None,
                   reasoning_effort=None, json_object=False, timeout=None,
                   use_max_completion_tokens=False):
    profile = capabilities(model, base_url)
    result = {'model': model, 'messages': messages}
    if temperature is not None and profile['temperature']:
        result['temperature'] = temperature
    # An unsupported capability is omitted entirely, including an explicit "none".
    if reasoning_effort is not None and profile['reasoning_efforts']:
        if reasoning_effort not in profile['reasoning_efforts']:
            raise ModelCapabilityError('Requested reasoning_effort is not supported by this model profile')
        result['reasoning_effort'] = reasoning_effort
    if output_tokens is not None:
        if type(output_tokens) is not int or output_tokens <= 0:
            raise ModelCapabilityError('Output token limit must be a positive integer')
        token_parameter = 'max_completion_tokens' if use_max_completion_tokens else profile['token_parameters'][0]
        if token_parameter not in profile['token_parameters']:
            raise ModelCapabilityError('Requested token parameter is not supported by this model profile')
        result[token_parameter] = output_tokens
    if json_object and profile['json_object']:
        result['response_format'] = {'type': 'json_object'}
    if timeout is not None:
        result['timeout'] = timeout
    if profile.get('extra_body'):
        result['extra_body'] = profile['extra_body']
    return result


def chat_completion(client, *, stage, **options):
    kwargs = request_kwargs(base_url=str(client.base_url), **options)
    audit = os.environ.get('AGENTDOJO_REQUEST_AUDIT')
    row = {'stage': stage, 'model': kwargs['model'], 'parameter_names': sorted(kwargs),
           'parameters': {k: kwargs[k] for k in ('temperature', 'reasoning_effort', 'max_tokens', 'max_completion_tokens', 'response_format', 'timeout') if k in kwargs}}
    # Capture only model parameters; never keys, messages, responses or headers.
    try:
        result = client.chat.completions.create(**kwargs)
    except Exception as exc:
        row.update(status='error', error_type=type(exc).__name__)
        raise
    else:
        row['status'] = 'ok'
        usage = getattr(result, 'usage', None)
        if usage is not None:
            row['usage'] = {k: getattr(usage, k, None) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        return result
    finally:
        if audit:
            path = Path(audit)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as stream:
                stream.write(json.dumps(row) + '\n')
