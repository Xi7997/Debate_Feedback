from pathlib import Path


class ReliabilityModel:
    def __init__(self, model_path, device=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if not Path(model_path).is_dir():
            raise ValueError(f"Assistant model directory does not exist: {model_path}")
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def score(self, case, comments):
        # Encode the case and comment as a pair, truncating the longer sequence first.
        inputs = self.tokenizer(
            [case] * len(comments),
            [f"Position: {c['label']}\nOpinion: {c['opinion']}" for c in comments],
            padding=True, truncation=True, max_length=512, return_tensors="pt",
        ).to(self.device)
        with self.torch.no_grad():
            scores = self.model(**inputs).logits.softmax(dim=-1)[:, 1]
        return scores.cpu().tolist()
