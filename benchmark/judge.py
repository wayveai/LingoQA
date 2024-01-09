import torch
from torch import nn

from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from constants import LINGO_JUDGE


class LingoJudge(nn.Module):
    """
    LingoJudge is a textual classifier that evaluates the truthfulness of an answer on the LingoQA benchmark.
    """
    def __init__(self, pretrained_model=LINGO_JUDGE):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
        #TODO: Update this with a single line once HF upload complete
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)

    def forward(self, question: str, reference: str, prediction: str):
        device = next(self.parameters()).device
        text = f"{self.tokenizer.cls_token}\nQuestion: {question}\nAnswer: {reference}\nStudent: {prediction}"
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        output = self.model(**encoded_input)

        return output.logits.squeeze(-1)

    def compute(self, question: str, references: List[str], prediction: str):
        """
        Computes the score for multiple reference associated with one prediction and returns the maximum score.

        Args:
            question: The reference question on which model predictions were made.
            references: A list of one of more correct reference answers.
            prediction: A single model prediction.
        """
        scores = []
        for reference in references:
            score = self.forward(question, reference, prediction)
            scores.extend(score)
        max_score = max(scores)
        return max_score
