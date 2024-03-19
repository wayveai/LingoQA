"""
LingoQA datasets are stored in Google Cloud.
This file provides the download link for the datasets, as well as reference keys for the data.
"""
from enum import Enum

# Example path to test set, to be updated upon dataset release
LINGOQA_TEST = "http://evaluation/test.parquet"

LINGO_JUDGE = "Lingo-Judge"

class Keys(str, Enum):
    question_id = "question_id"
    segment_id = "segment_id"
    question = "question"
    answer = "answer"
    references = "references"
    prediction = "prediction"
    max_score = "max_score"
    score = "score"
    probability = "probability"
    correct = "correct"