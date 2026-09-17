import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "skills/agentdojo-annotation/scripts/task_alignment"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("task_annotator", SCRIPTS / "annotate_deterministic.py")
annotator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annotator)


def action(recipient="alice@example.com", result=True, error="null", body="hello", **extra):
    args = {"to": recipient, "body": body, **extra}
    text = "[ACTION RESULT PAIR]\n[ACTION]\nfunction: send_email\nargs: " + json.dumps(args)
    if result:
        text += f"\n[RESULT]\nerror: {error}\ncontent: sent"
    else:
        text = text.replace("[ACTION RESULT PAIR]", "[ASSISTANT TOOL CALL]")
    return {"role": "assistant", "content": text}


class TaskConstraints(unittest.TestCase):
    def annotate(self, actions, prompt="Send an email to alice@example.com saying hello.", specification=None):
        return annotator.annotate_trajectory(
            {"suite_name": "workspace", "trajectory": [{"role": "user", "content": prompt}, *actions]},
            Path("synthetic.json"), task_spec=specification,
        )

    def test_later_wrong_recipient_does_not_replace_requested_action(self):
        record, reason = self.annotate([action(), action("mallory@example.com")])
        self.assertIsNone(reason)
        self.assertEqual(record["target"]["component_id"], "C2")

    def test_unknown_or_failed_execution_is_not_selected(self):
        for candidate in [action(result=False), action(error="unknown"), action(error="timeout"),
                          {"role": "assistant", "content": action()["content"].replace("error: null", "no_error_field: true")},
                          {"role": "assistant", "content": action()["content"].replace("content: sent", 'content: {"success": false}')}]:
            with self.subTest(candidate=candidate):
                record, reason = self.annotate([candidate])
                self.assertIsNone(record)
                self.assertEqual(reason, "no_successful_requested_core_action")

    def test_wrong_content_and_extra_recipients_do_not_match(self):
        for candidate in [action(body="different"), action(bcc=["mallory@example.com"]), action(["alice@example.com", "mallory@example.com"])]:
            self.assertIsNone(self.annotate([candidate])[0])

    def test_unknown_or_negated_prompt_is_not_guessed(self):
        for prompt in ["Do not send an email to alice@example.com saying hello.", "Please handle my email."]:
            record, reason = self.annotate([action()], prompt)
            self.assertIsNone(record)
            self.assertEqual(reason, "unverified_task_constraints")

    def test_subject_constraint(self):
        prompt = 'Send an email to alice@example.com with subject "Update" and body "hello".'
        self.assertIsNotNone(self.annotate([action(subject="Update")], prompt)[0])
        self.assertIsNone(self.annotate([action(subject="Wrong")], prompt)[0])

    def test_explicit_constraints_and_prompt_binding(self):
        prompt = "Please handle my email."
        constraint = {"prompt": prompt, "actions": [{"function": "send_email", "args": {"to": "alice@example.com", "body": "hello"}}]}
        self.assertIsNotNone(self.annotate([action()], prompt, constraint)[0])
        self.assertIsNone(self.annotate([action(bcc=["mallory@example.com"])], prompt, constraint)[0])
        with self.assertRaises(ValueError):
            self.annotate([action()], "Different prompt", constraint)

    def test_raw_pipeline_and_external_constraints(self):
        repository = SCRIPTS.parents[3]
        fixture = json.loads((repository / "examples/task-alignment/raw/workspace/example.json").read_text())
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw/workspace"; raw.mkdir(parents=True)
            fixture["messages"][0]["content"] = "Please notify Alice about the release."
            (raw / "example.json").write_text(json.dumps(fixture))
            constraints = root / "constraints.json"
            constraints.write_text(json.dumps({"workspace/example.json": {
                "prompt": fixture["messages"][0]["content"],
                "actions": [{"function": "send_email", "args": {"to": "alice@example.com", "body": "hello"}}],
            }}))
            for configured in [False, True]:
                output = root / str(configured)
                command = [sys.executable, str(SCRIPTS / "run_pipeline.py"), "--raw_root", str(raw.parent), "--output_root", str(output), "--write_skipped"]
                if configured:
                    command.extend(["--task_constraints", str(constraints)])
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0 if configured else 1, result.stderr)
                manifest = json.loads((output / "run_manifest.json").read_text())
                self.assertEqual(manifest["status"], "ok" if configured else "completed_with_review")
                if configured:
                    annotation = json.loads((output / "annotations/workspace/example.json").read_text())
                    self.assertEqual(annotation["target"]["args"]["to"], "alice@example.com")
                else:
                    self.assertTrue((output / "annotations/_skipped_annotations.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
