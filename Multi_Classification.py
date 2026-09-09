import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_utils import load_task
from prompts import judge_prompt, debater_prompt, EXCHANGE_PROMPT, FINAL_PROMPT


METHODS = {"debate-feedback": "debate-single", "assistant_debate": "debate-assistant"}


def parse_json(content):
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("Model output must be a JSON object")
    return result


def normalize_probabilities(result, labels):
    probabilities = result["probabilities"]
    if set(probabilities) != set(labels):
        raise ValueError("Judge output must contain exactly the allowed labels")
    values = [float(probabilities[label]) for label in labels]
    if any(not math.isfinite(x) or x < 0 for x in values) or sum(values) <= 0:
        raise ValueError("Judge output contains invalid probabilities")
    total = sum(values)
    return {label: value / total for label, value in zip(labels, values)}


def smooth(previous, current, weight):
    return {label: (1 - weight) * previous[label] + weight * current[label] for label in previous}


class AdversarialAI:
    def __init__(self, args, client=None, classifier=None):
        self.args = args
        self.data, self.labels = load_task(args.task)
        self.rows = self.data[args.split]
        if args.limit:
            self.rows = self.rows[:args.limit]
        if client is None:
            from openai import OpenAI
            client = OpenAI(max_retries=3, timeout=120)
        self.client = client
        self.classifier = classifier
        self.assistant_lock = threading.Lock()
        if args.method == "debate-assistant" and classifier is None:
            from assistant import ReliabilityModel
            metadata = Path(args.assistant_model) / "training_config.json"
            if metadata.exists():
                with metadata.open(encoding="utf-8") as f:
                    if json.load(f)["task"] != args.task:
                        raise ValueError("Assistant training task does not match the current task")
            self.classifier = ReliabilityModel(args.assistant_model)

    def request(self, history, validator):
        # The SDK retries network errors; retry invalid output structures here.
        messages = list(history)
        for attempt in range(3):
            response = self.client.chat.completions.create(
                model=self.args.model, messages=messages, temperature=self.args.temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            try:
                result = validator(parse_json(content))
                history.append({"role": "assistant", "content": content})
                return result
            except (ValueError, KeyError, TypeError) as error:
                if attempt == 2:
                    raise ValueError(f"Failed to parse model output: {error}") from error
                messages.extend([
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "Invalid output. Follow the required JSON schema and allowed labels exactly."},
                ])

    def judge(self, history):
        return self.request(history, lambda result: normalize_probabilities(result, self.labels))

    def debate(self, history):
        def validate(result):
            label = str(result["label"])
            if label not in self.labels or not isinstance(result["opinion"], str) or not result["opinion"].strip():
                raise ValueError("Invalid expert position or opinion")
            return {"label": label, "opinion": result["opinion"]}
        return self.request(history, validate)

    def single_debate(self, row):
        args = self.args
        judge_history = [
            {"role": "system", "content": judge_prompt(args.task, self.labels)},
            {"role": "user", "content": row["text"]},
        ]
        initial = self.judge(judge_history)
        previous = initial
        rounds = []
        histories = [[
            {"role": "system", "content": debater_prompt(args.task, self.labels)},
            {"role": "user", "content": row["text"]},
        ] for _ in range(args.debaters or 0)]
        for i in range(0 if args.method == "zeroshot" else args.rounds):
            opinions = []
            for history in histories:
                if i:
                    history.append({"role": "user", "content": "Begin the next round. Reconsider the case and state your current view."})
                opinions.append(self.debate(history))
            comments = []
            for j, history in enumerate(histories):
                others = [dict(expert=k, **opinion) for k, opinion in enumerate(opinions) if k != j]
                history.append({"role": "user", "content": EXCHANGE_PROMPT + json.dumps(others, ensure_ascii=False)})
                comments.append(self.debate(history))
            reliability = None
            accepted = True
            if self.classifier is not None:
                with self.assistant_lock:
                    reliability = self.classifier.score(row["text"], comments)
                # Accept the round based on the mean reliability of expert comments.
                accepted = sum(reliability) / len(reliability) >= args.threshold
            feedback = {"opinions": opinions, "comments": comments, "reliability": reliability}
            judge_history.append({"role": "user", "content": FINAL_PROMPT + json.dumps(feedback, ensure_ascii=False)})
            current = self.judge(judge_history)
            # Fall back to the initial prediction on low reliability; otherwise apply exponential smoothing.
            previous = smooth(previous, current, args.smoothing) if accepted else dict(initial)
            rounds.append(dict(round=i + 1, **feedback, accepted=accepted,
                               judge_probabilities=current, smoothed_probabilities=previous))
        return dict(row, status="ok", initial_probabilities=initial,
                    initial_prediction=max(initial, key=initial.get), rounds=rounds,
                    prediction=max(previous, key=previous.get), probabilities=previous)

    def run(self):
        args = self.args
        output = Path(args.output or f"Output/{args.task}_{args.method}_{args.split}.jsonl")
        output.parent.mkdir(parents=True, exist_ok=True)
        config = {key: value for key, value in vars(args).items() if key not in ("output", "resume", "workers", "limit")}
        config["labels"] = self.labels
        config["api_base"] = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        done = set()
        if output.exists():
            if not args.resume:
                raise ValueError(f"Output already exists; choose another path or use --resume: {output}")
            with output.open(encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    if record["config"] != config:
                        raise ValueError("Cannot resume with a different experiment configuration")
                    if record["status"] == "ok":
                        done.add((record["id"], record["text_hash"]))
        rows = [row for row in self.rows if (row["id"], hashlib.sha256(row["text"].encode()).hexdigest()) not in done]
        failures = 0
        with output.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(self.single_debate, row): row for row in rows}
            for count, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    failures += 1
                    record = dict(row, status="error", error=str(error))
                record.update(config=config, text_hash=hashlib.sha256(row["text"].encode()).hexdigest())
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{count}/{len(rows)}] case={row['id']} status={record['status']}")
        print(f"Results saved to {output}; {failures} failures in this run. Use --resume to retry failed cases.")


def main():
    parser = argparse.ArgumentParser(description="Debate-Feedback legal judgment prediction")
    parser.add_argument("--task", choices=["trial", "article", "crime"], default="trial")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--method", choices=["zeroshot", "debate-single", "debate-assistant"] + list(METHODS), default="zeroshot")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--rounds", type=int, help="Number of rounds; required for debate methods")
    parser.add_argument("--debaters", type=int, help="Number of experts; required for debate methods")
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--assistant-model")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 uses all samples")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.method = METHODS.get(args.method, args.method)
    if args.method != "zeroshot" and (args.rounds is None or args.debaters is None):
        parser.error("Debate methods require --rounds and --debaters")
    if ((args.rounds is not None and args.rounds < 1)
            or (args.debaters is not None and args.debaters < 2)
            or args.workers < 1 or args.limit < 0):
        parser.error("rounds/workers must be positive, debaters at least 2, and limit nonnegative")
    if not 0 <= args.smoothing <= 1 or not 0 <= args.threshold <= 1:
        parser.error("smoothing and threshold must be in [0, 1]")
    if args.method == "debate-assistant" and not args.assistant_model:
        parser.error("debate-assistant requires --assistant-model")
    AdversarialAI(args).run()


if __name__ == "__main__":
    main()
