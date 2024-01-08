import torch
from torch import nn

from typing import List
from transformers import AutoModel, AutoTokenizer
from constants import LINGO_JUDGE


class LingoJudge(nn.Module):
    """
    LingoJudge is a textual classifier that evaluates the truthfulness of an answer.
    """
    def __init__(self, pretrained_model=LINGO_JUDGE):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
        self.model = AutoModel.from_pretrained(pretrained_model)
        self.head = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, question: str, reference: str, prediction: str):
        device = next(self.parameters()).device
        text = f"{self.tokenizer.cls_token}\nQuestion: {question}\nAnswer: {reference}\nStudent: {prediction}"
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        output = self.model(**encoded_input)
        embedding = output.last_hidden_state[:, 0]
        return self.head(embedding).squeeze(-1)

    def compute(self, question: str, references: List[str], prediction: str):
        """
        Computes the score for multiple reference associated with one prediction and returns the maximum score.

        Args:
            question: The reference question on which model predictions were made.
            prediction: A single model prediction.
            references: A list of one of more correct reference answers.
        """
        scores = []
        for reference in references:
            score = self.forward(question, reference, prediction)
            scores.append(score)
        max_score = max(scores)
        return max_score
