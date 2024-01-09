"""
LingoQA datasets are stored in Google Cloud.
This file provides the download link for the datasets, as well as reference keys for the data.
"""
from enum import Enum

#TODO: Update this with actual datasets once uploaded on Google Drive
LINGOQA_TEST = "http://172.24.42.122:8080/test_export/val.parquet"

#TODO: Update this with the file name from Long
LINGO_JUDGE = "microsoft/deberta-v3-base"

class Keys(str, Enum):
    question_id = "question_id"
    segment_id = "segment_id"
    question = "question"
    answer = "answer"
    references = "references"
    prediction = "prediction"
    max_score = "max_score"
    probability = "probability"
    correct = "correct"

