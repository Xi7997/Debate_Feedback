import json


TASK_DESCRIPTION = {
    "trial": "Predict the court outcome: Plaintiff wins or Defendant wins.",
    "article": "Predict the single relevant article of the Criminal Law of the People's Republic of China.",
    "crime": "Predict the criminal charge under Chinese criminal law.",
}


def judge_prompt(task, labels):
    return (
        "You are a judge analyzing a legal case. " + TASK_DESCRIPTION[task]
        + f" Allowed labels: {json.dumps(labels, ensure_ascii=False)}. "
        'Return ONLY a JSON object: {"probabilities": {"label": probability, ...}}. '
        "Include every allowed label exactly once, with nonnegative probabilities summing to 1."
    )


def debater_prompt(task, labels):
    return (
        "You are a legal expert debating a case with other experts. " + TASK_DESCRIPTION[task]
        + f" Allowed labels: {json.dumps(labels, ensure_ascii=False)}. "
        "State your position and support it with facts from the case. Do not invent facts. "
        'Return ONLY JSON: {"label": "one allowed label", "opinion": "at most 200 words"}.'
    )


EXCHANGE_PROMPT = (
    "Read the other experts' opinions below. You may agree or counter-argue. "
    "Express your current position clearly for the judge, using the same JSON format.\n"
)
FINAL_PROMPT = (
    "Review this round's arguments and comments. Reliability, when supplied, estimates "
    "whether an expert's current label is correct. Provide your updated judgment using "
    "the same probability JSON format.\n"
)
