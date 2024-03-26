import click
import torch
import pandas as pd

from datasets import Dataset
from functools import partial 
from constants import LINGOQA_TEST, Keys
from judge import LingoJudge


@click.command()
@click.option('--predictions_path', help='Path to predictions file.')
def evaluate(predictions_path: str) -> float:
    """
    Simple script for running evaluation on the LingoQA benchmark.

    Args:
        predictions_path: path to a .csv file containing the model predictions.
    Out:
        benchmark_score: evaluation score obtained from running the textual classifier on the benchmark.
    """
    references = pd.read_parquet(LINGOQA_TEST)
    references = references[[Keys.question_id, Keys.segment_id, Keys.question, Keys.answer]]
    references = references.groupby([Keys.question_id, Keys.segment_id, Keys.question]).agg(list)
    references = references.rename({Keys.answer: Keys.references}, axis=1)
    print(f"Loaded {len(references)} references.")

    predictions = pd.read_csv(predictions_path)
    predictions = predictions.rename({Keys.answer: Keys.prediction}, axis=1)
    print(f"Loaded {len(predictions)} predictions.")
     
    merged = pd.merge(predictions, references, on=[Keys.question_id, Keys.segment_id])
    print(f"Matched {len(merged)} predictions with references.")
    if len(merged) != 500:
        print("WARNING! You are evaluating on a subset of the LingoQA benchmark. Please check your input file for missing or mis-matched examples.")

    dataset = Dataset.from_pandas(merged)

    judge = LingoJudge().eval().to("cuda:0")
    dataset_evaluated = dataset.map(partial(evaluate_question, judge), batched=True)
    dataset_filtered = dataset_evaluated.filter(select_correct)

    benchmark_score = dataset_filtered.num_rows/dataset_evaluated.num_rows
    print(f"The overall benchmark score is {benchmark_score*100}%")
    return benchmark_score


def evaluate_question(metric: LingoJudge, data_dict: dict) -> dict:
    """
    Run evaluation for one question.

    Args:
        metric: the evaluation metric for computing the scores.
        data_dict: the data dictionary containing a question, a reference, and a prediction.

    Out:
        data_dict: updated data dictionary containing information such as
        the maximum score, the probability of correctness, and a boolean
        indicating whether the prediction is correct or not.
    """
    questions = data_dict[Keys.question]
    references = data_dict[Keys.references]
    prediction = data_dict[Keys.prediction]

    assert len(data_dict[Keys.references]) > 1 and isinstance(data_dict[Keys.references], list), f"Expected a list of more than one reference answer per question, but got: {data_dict[Keys.references]}"

    scores = metric.compute(questions, references, prediction)
    data_dict[Keys.score] = scores
    data_dict[Keys.probability] = torch.sigmoid(scores)
    data_dict[Keys.correct] = scores > 0.0
    return data_dict


def select_correct(data_dict: dict) -> bool:
    """
    Filtering function for selecting the predictions classified as correct.
    """
    return data_dict[Keys.correct]


if __name__=="__main__":
    _ = evaluate()