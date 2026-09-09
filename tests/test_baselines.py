import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from Eval.common import prepare_data, default_model


ROOT = Path(__file__).resolve().parents[1]


class BaselineTests(unittest.TestCase):
    def test_all_entrypoints(self):
        for name in ['CNN', 'LegalBert', 'Lawformer']:
            result = subprocess.run([sys.executable, str(ROOT / 'Eval' / f'{name}.py'), '--help'],
                                    cwd='/tmp', capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--task', result.stdout)
            self.assertIn('--max-length', result.stdout)

    def test_paper_model_defaults(self):
        self.assertEqual(default_model('legal_bert', 'trial'), 'nlpaueb/legal-bert-base-uncased')
        self.assertEqual(default_model('lawformer', 'article'), 'thunlp/Lawformer')
        self.assertEqual(default_model('cnn', 'trial'), 'bert-base-uncased')
        self.assertEqual(default_model('cnn', 'article'), 'bert-base-chinese')

    def test_data_and_saved_mapping(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            for task in ['trial', 'article', 'crime']:
                args = argparse.Namespace(task=task, limit=3, output=str(Path(folder) / task))
                data, texts, targets, labels, output = prepare_data(args)
                for split in ['train', 'valid', 'test']:
                    self.assertEqual(len(texts[split]), 3)
                    for row, index in zip(data[split], targets[split]):
                        self.assertEqual(row['label'], labels[index])
                self.assertEqual(json.loads((output / 'labels.json').read_text()), labels)
                with self.assertRaises(ValueError):
                    prepare_data(args)

    def test_debate_parameters_required(self):
        result = subprocess.run([sys.executable, str(ROOT / 'Multi_Classification.py'),
                                 '--method', 'debate-single'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('require --rounds and --debaters', result.stderr)


if __name__ == '__main__':
    unittest.main()
