from datasets import Dataset
from functools import partial 

import torch
import pandas as pd
from constants import LINGOQA_TEST, LINGO_JUDGE
from judge import LingoJudge

#TODO: Should we do it in batches such that it is a little bit faster?
def evaluate(predictions="./predictions.csv"):
    references = pd.read_parquet(LINGOQA_TEST)
    assert len(references) == 1000
    references = references[["question_id", "segment_id", "question", "answer"]]
    references = references.groupby(["question_id", "segment_id", "question"]).agg(list)
    references = references.rename({"answer": "references"}, axis=1)

    predictions = pd.read_csv(predictions)
    assert len(predictions) == 500
    predictions = predictions.rename({"answer": "prediction"}, axis=1)

    merged = pd.merge(predictions, references, on=["question_id", "segment_id"])
    assert len(merged) == 500

    dataset = Dataset.from_pandas(merged)
    
    judge = LingoJudge()

    dataset.map(partial(evaluate_question, judge))

    benchmark_score = 0
    return benchmark_score


def evaluate_question(metric: LingoJudge, data_dict: dict):
    """
    Runs evaluation for one question in batch.
    """
    questions = data_dict['question']
    references = data_dict['references']
    prediction = data_dict['prediction']

    max_score = metric.compute(questions, references, prediction)

    data_dict['max_score'] = max_score
    data_dict['correct'] = max_score > 0.0
    data_dict['prob'] = torch.sigmoid(max_score)

    return data_dict


if __name__=="__main__":
    evaluate()