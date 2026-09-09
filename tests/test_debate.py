import argparse
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from Multi_Classification import AdversarialAI, normalize_probabilities
from data_utils import load_task
from Output.compute_acc import evaluate
from train_assistant import load_examples


class FakeClient:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        answer = next(self.answers)
        content = answer if isinstance(answer, str) else json.dumps(answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class DebateTests(unittest.TestCase):
    def args(self, **kwargs):
        options = dict(task='trial', split='test', limit=1, method='debate-single',
                       model='fake', temperature=0, rounds=2, debaters=2,
                       smoothing=0.5, threshold=0.5, assistant_model=None,
                       workers=1, output=None, resume=False)
        options.update(kwargs)
        return argparse.Namespace(**options)

    def answers(self):
        opinion = {'label': 'Plaintiff wins', 'opinion': 'The case supports the plaintiff.'}
        initial = {'probabilities': {'Defendant wins': 0.8, 'Plaintiff wins': 0.2}}
        updated = {'probabilities': {'Defendant wins': 0.1, 'Plaintiff wins': 0.9}}
        return [initial] + [opinion] * 4 + [updated] + [opinion] * 4 + [updated]

    def test_multiround_smoothing(self):
        runner = AdversarialAI(self.args(), client=FakeClient(self.answers()))
        row = runner.single_debate(runner.rows[0])
        self.assertEqual(row['initial_prediction'], 'Defendant wins')
        self.assertEqual(row['prediction'], 'Plaintiff wins')
        self.assertAlmostEqual(row['probabilities']['Plaintiff wins'], 0.725)
        self.assertEqual(len(row['rounds'][0]['comments']), 2)

    def test_comments_are_not_initial_opinions(self):
        answers = self.answers()
        answers[3] = {'label': 'Defendant wins', 'opinion': 'A revised counter-argument.'}
        runner = AdversarialAI(self.args(rounds=1), client=FakeClient(answers))
        row = runner.single_debate(runner.rows[0])
        self.assertEqual(row['rounds'][0]['comments'][0]['opinion'], 'A revised counter-argument.')
        self.assertNotEqual(row['rounds'][0]['comments'][0], row['rounds'][0]['opinions'][0])

    def test_reliability_gate(self):
        class LowReliability:
            def score(self, case, comments):
                return [0.1, 0.2]
        runner = AdversarialAI(self.args(method='debate-assistant'),
                               client=FakeClient(self.answers()), classifier=LowReliability())
        row = runner.single_debate(runner.rows[0])
        self.assertEqual(row['probabilities'], row['initial_probabilities'])
        self.assertFalse(row['rounds'][1]['accepted'])

    def test_json_retry_and_zeroshot(self):
        runner = AdversarialAI(self.args(method='zeroshot'),
                               client=FakeClient(['invalid'] + self.answers()))
        row = runner.single_debate(runner.rows[0])
        self.assertEqual(row['rounds'], [])
        self.assertEqual(row['prediction'], row['initial_prediction'])
        with self.assertRaises(ValueError):
            normalize_probabilities({'probabilities': {'x': float('nan')}}, ['x'])

    def test_splits(self):
        for task in ['trial', 'article', 'crime']:
            data, labels = load_task(task)
            train = {r['text'] for r in data['train']}
            valid = {r['text'] for r in data['valid']}
            test = {r['text'] for r in data['test']}
            self.assertFalse(train & (valid | test))
            self.assertTrue(labels)
        self.assertEqual(len(load_task('trial')[0]['train']), 7591)

    def test_output_resume_and_training_examples(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'results.jsonl')
            args = self.args(split='train', output=path)
            runner = AdversarialAI(args, client=FakeClient(self.answers()))
            runner.run()
            examples = load_examples(path, 'train', 'trial')
            self.assertEqual(len(examples), 4)
            self.assertEqual(examples[0][2], int(runner.rows[0]['label'] == 'Plaintiff wins'))
            args.resume = True
            # No responses remain; resuming must not request another prediction.
            with patch.object(runner, 'single_debate', side_effect=AssertionError('Must not repeat completed requests')) as call:
                runner.run()
                call.assert_not_called()
            self.assertEqual(evaluate(path)['total'], 1)
            with self.assertRaises(ValueError):
                load_examples(path, 'valid', 'trial')

    def test_evaluation_counts_failures_and_deduplicates(self):
        records = [
            dict(id=1, text_hash='a', config={}, status='error', label='A'),
            dict(id=1, text_hash='a', config={}, status='ok', label='A', prediction='A', initial_prediction='B'),
            dict(id=2, text_hash='b', config={}, status='error', label='B'),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'results.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in records))
            result = evaluate(path)
            self.assertEqual(result['total'], 2)
            self.assertEqual(result['failed'], 1)
            self.assertEqual(result['accuracy'], 0.5)
            self.assertEqual(result['macro_f1'], 0.5)
            self.assertEqual(result['prediction_correction'], 1)


if __name__ == '__main__':
    unittest.main()
