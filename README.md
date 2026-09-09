# Debate-Feedback

Implementation of *Debate-Feedback: A Multi-Agent Framework for Efficient Legal Judgment Prediction*. The framework combines an initial judgment, multi-agent debate, assistant-based reliability evaluation, and prediction smoothing.

## Repository structure

- `Multi_Classification.py`: entry point for `zeroshot`, `debate-single` (without an assistant), and `debate-assistant`.
- `prompts.py`: judge, expert, and opinion-exchange prompts. Experts can choose and revise their positions.
- `data_utils.py`: dataset loading and label normalization.
- `assistant.py`, `train_assistant.py`: reliability model inference and training.
- `Output/compute_acc.py`: accuracy, macro-F1, initial accuracy, and prediction correction/degradation counts.
- `Eval/`: CNN with BERT, Legal-BERT, and Lawformer baselines with shared training and evaluation utilities.
- `Dataset/`: trial outcome (`trial`), article (`article`), and charge (`crime`) prediction datasets. Files ending in `.json` contain one JSON record per line.
- `Dataset/Original/`: raw case examples, extracted text, and a standalone case summarization script.

## Setup and basic usage

Use Python 3.10+ and run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
# Optional: export OPENAI_BASE_URL="your-compatible-api-url"

# Small runs; omit --limit to process the entire selected split.
python Multi_Classification.py --task trial --method zeroshot --limit 10
python Multi_Classification.py --task article --method debate-single --rounds 2 --debaters 2 --limit 10
python Output/compute_acc.py Output/article_debate-single_test.jsonl
```

The default model is `gpt-4o-mini`. Table 2 uses GPT-4o for the judge and debaters; Table 3 uses GPT-4o mini. Select an API-supported model with `--model`. The endpoint must support Chat Completions, JSON output, and the temperature parameter.

```bash
python Multi_Classification.py --task trial --method debate-single \
  --model gpt-4o --rounds 2 --debaters 2 --smoothing 0.5 \
  --workers 2 --limit 20 --output Output/trial_single.jsonl
```

Results include case text, labels, expert positions and comments, reliability scores, and judge probabilities for each round. Existing files are not overwritten. Add `--resume` to the same command to skip successful cases and retry failures; `--limit` may be increased. Use a new output path when changing the experiment configuration.

Evaluation deduplicates records. Failed cases count toward the denominator as missing predictions; correction and degradation counts use successful cases only.

## Assistant model

Generate unassisted debate records on the training and validation splits, then train a binary classifier to predict whether an expert's position is correct from the case and comment. The limits below are for small runs; use more samples for training. Training comments must include both correct and incorrect positions.

```bash
python Multi_Classification.py --task trial --split train --method debate-single \
  --rounds 2 --debaters 2 --limit 200 --output Output/trial_train.jsonl
python Multi_Classification.py --task trial --split valid --method debate-single \
  --rounds 2 --debaters 2 --limit 100 --output Output/trial_valid.jsonl
python train_assistant.py --task trial --train Output/trial_train.jsonl \
  --valid Output/trial_valid.jsonl --output assistant_model/trial
python Multi_Classification.py --task trial --method debate-assistant \
  --rounds 2 --debaters 2 --assistant-model assistant_model/trial --threshold 0.5 --limit 20
python Output/compute_acc.py Output/trial_debate-assistant_test.jsonl
```

Train a separate assistant using records from the corresponding task. The default encoder is `bert-base-multilingual-cased` with a binary classification head; select another encoder with the training script's `--model` option. The first run downloads pretrained weights. The best checkpoint is selected by validation macro-F1. Choose the reliability threshold on validation data, not the test set.

## Runtime parameters

- `--debaters` and `--rounds` set the number of experts and debate rounds. Both are required for debate methods and optional for zero-shot prediction. `single` means no assistant, not one round.
- The judge returns normalized class probability scores. `--smoothing T` controls `O_i = (1 - T) * O_(i-1) + T * P_i`. The default is `T=0.5`; use 1 to disable smoothing.
- The assistant estimates the probability that an expert's position is correct. A round is accepted when mean reliability reaches `--threshold`; otherwise, the prediction returns to the initial probabilities.
- Training excludes texts present in validation or test data. Trial prediction is binary; article and charge prediction select one class from multiple candidates. Candidate labels are defined in `Dataset/labels.json`.

## Baselines

The three supervised baselines correspond to CNN with BERT, Legal-BERT, and Lawformer in Table 2. They share `--task`, `--model`, `--epochs`, `--batch-size`, `--max-length`, `--lr`, `--limit`, `--seed`, and `--output`. The limit applies to each split; the default uses all samples.

```bash
python Eval/CNN.py --task article --epochs 5 --output Output/cnn_article
python Eval/LegalBert.py --task trial --epochs 5 --output Output/legal_bert_trial
python Eval/Lawformer.py --task article --epochs 5 --max-length 4096 \
  --batch-size 2 --output Output/lawformer_article
```

- `CNN.py`: BERT token representations, convolution windows of sizes 3/4/5 with 128 channels each, max pooling, and a classification layer. BERT is frozen by default; enable joint fine-tuning with `--finetune-bert`. Defaults to `bert-base-uncased` for English and `bert-base-chinese` for Chinese tasks.
- `LegalBert.py`: fine-tunes [Legal-BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased), using `nlpaueb/legal-bert-base-uncased`.
- `Lawformer.py`: fine-tunes [Lawformer](https://huggingface.co/thunlp/Lawformer), using `thunlp/Lawformer`, with global attention on the classification token.

All three support `trial`, `article`, and `crime`. Legal-BERT uses English legal pretrained weights; Lawformer uses Chinese legal pretrained weights. These weights remain unchanged when selecting another task unless overridden with `--model`. All baselines use PyTorch; TensorFlow is not required.

The best checkpoint is selected by validation macro-F1. Each output directory contains the model, tokenizer, label mapping, configuration, test predictions, and `metrics.json` with accuracy, macro-F1, and top-5/top-10 accuracy. When fewer than k classes exist, top-k includes every class. Use separate output directories for different runs.

Run the GPT-3.5-turbo and GPT-4o zero-shot baselines from Table 2 with the main script:

```bash
python Multi_Classification.py --task trial --method zeroshot --model gpt-3.5-turbo \
  --output Output/trial_gpt35.jsonl
python Multi_Classification.py --task trial --method zeroshot --model gpt-4o \
  --output Output/trial_gpt4o.jsonl
```

Model names must be supported by your API service. For article prediction, use `--task article` and separate output paths.

## Offline checks

These tests do not call the API or download model weights:

```bash
python -B -m unittest discover -s tests -v
```
