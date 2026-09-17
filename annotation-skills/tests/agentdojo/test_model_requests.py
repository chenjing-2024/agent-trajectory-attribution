"""Exercise real request call sites without network or model-generated labels."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / 'skills/agentdojo-annotation/scripts/unsafe'
sys.path.insert(0, str(SCRIPTS))
import model_requests as adapter

def module(name):
    spec = importlib.util.spec_from_file_location('adapter_test_' + name, SCRIPTS / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

annotate, semantic, repair = [module(x) for x in ('annotate', 'validate_semantic', 'repair')]

class FakeClient:
    base_url = 'https://api.openai.com/v1/'
    def __init__(self, responses=('{}',), error=None):
        self.calls = []
        self.responses = iter(responses)
        self.error = error
        self.chat = SimpleNamespace(completions=self)
    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(self.responses)), finish_reason='stop')], usage=None)

class RequestTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def kwargs(self, **changes):
        options = dict(model='gpt-4.1-mini', base_url=FakeClient.base_url, messages=[], temperature=0, output_tokens=4096, reasoning_effort='medium', json_object=True)
        options.update(changes)
        return adapter.request_kwargs(**options)

    def test_non_reasoning_omits_even_none(self):
        for effort in ('low', 'medium', 'none'):
            request = self.kwargs(reasoning_effort=effort)
            self.assertNotIn('reasoning_effort', request)
            self.assertNotIn('extra_body', request)
            self.assertEqual(request['max_tokens'], 4096)
            self.assertEqual(request['temperature'], 0)
        self.assertEqual(self.kwargs(use_max_completion_tokens=True)['max_completion_tokens'],4096)

    def test_reasoning_model_preserves_supported_effort(self):
        request = self.kwargs(model='gpt-5.4-mini')
        self.assertEqual(request['reasoning_effort'], 'medium')
        self.assertEqual(request['max_completion_tokens'], 4096)
        self.assertNotIn('temperature', request)
        with self.assertRaises(adapter.ModelCapabilityError):
            self.kwargs(model='gpt-5.4-mini', reasoning_effort='invented')

    def test_unknown_model_or_provider_requires_profile(self):
        for changes in ({'model':'unknown'}, {'base_url':'https://api.openai.com.example/v1'}, {'base_url':'http://localhost:8000/v1'}):
            with self.assertRaises(adapter.ModelCapabilityError):
                self.kwargs(**changes)

    def test_explicit_provider_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'capabilities.json'
            profile = dict(base_url='http://localhost:8000/v1', token_parameters=['max_tokens'], temperature=True, reasoning_efforts=[], json_object=False, extra_body={'chat_template_kwargs':{'enable_thinking':False}})
            path.write_text(json.dumps({'models':{'local-model':profile}}))
            os.environ['AGENTDOJO_MODEL_CAPABILITIES'] = str(path)
            request = self.kwargs(model='local-model', base_url=profile['base_url'])
            self.assertIn('extra_body', request)
            self.assertNotIn('response_format', request)
            self.assertNotIn('reasoning_effort', request)
            with self.assertRaises(adapter.ModelCapabilityError):
                self.kwargs(model='local-model')

    def test_initial_annotation_and_json_repair(self):
        client = FakeClient(('not json','{}'))
        with patch.object(annotate, 'validate_annotation', side_effect=lambda value,*args:value):
            result, _ = annotate.call_model(client, 'gpt-4.1-mini', 'test', [], 'case')
        self.assertEqual(result,{})
        self.assertEqual(len(client.calls),2)
        self.assertTrue(all('reasoning_effort' not in x for x in client.calls))

    def test_semantic_validation(self):
        client = FakeClient()
        result = semantic.call_validator(client, model='gpt-4.1-mini', messages=[], temperature=0, max_tokens=4096, max_retries=0, sleep_seconds=0)
        self.assertEqual(result,{})
        self.assertNotIn('reasoning_effort',client.calls[0])
        self.assertEqual(client.calls[0]['max_tokens'],4096)

    def test_repair_and_json_repair(self):
        client = FakeClient(('not json','{}'))
        with patch.object(repair, 'validate_annotation_structure', side_effect=lambda value,*args:value):
            result, _, repaired, _ = repair.call_model(client,model='gpt-4.1-mini',prompt='test',components=[],trajectory_id='case',reasoning_effort='medium',max_completion_tokens=4096,request_timeout=90,max_retries=0)
        self.assertEqual(result,{})
        self.assertEqual(repaired,'{}')
        self.assertEqual(len(client.calls),2)
        for call in client.calls:
            self.assertNotIn('reasoning_effort',call)
            self.assertEqual(call['max_tokens'],4096)
            self.assertEqual(call['response_format'],{'type':'json_object'})

    def test_api_error_is_preserved_without_parameter_fallback(self):
        error=RuntimeError('quota exhausted')
        client=FakeClient(error=error)
        with self.assertRaises(RuntimeError) as caught:
            adapter.chat_completion(client,stage='test',model='gpt-4.1-mini',messages=[],reasoning_effort='medium')
        self.assertIs(caught.exception,error)
        self.assertEqual(len(client.calls),1)

    def test_audit_contains_parameters_not_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'audit.jsonl'
            os.environ['AGENTDOJO_REQUEST_AUDIT']=str(path)
            adapter.chat_completion(FakeClient(),stage='test',model='gpt-4.1-mini',messages=[{'role':'user','content':'PRIVATE_SENTINEL'}],reasoning_effort='medium')
            text=path.read_text()
            self.assertNotIn('PRIVATE_SENTINEL',text)
            row=json.loads(text)
            self.assertEqual(row['status'],'ok')
            self.assertNotIn('reasoning_effort',row['parameter_names'])

if __name__=='__main__':
    unittest.main()
