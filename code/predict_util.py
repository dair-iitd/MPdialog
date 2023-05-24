import re
import numpy as np
import pandas as pd
import logging
from transformers import TrainingArguments
from transformers.trainer_utils import PredictionOutput
from torch.utils.data import Dataset

import os
import sys
import json
from params import MetricHparams
from params import GenerationHparams
from string import punctuation
logger = logging.getLogger(__name__)


class PostProcessors:
    @staticmethod
    def first_sentence(x):
        stop_pos = x.find(".")
        return x[: stop_pos + 1] if stop_pos != -1 else x

    @staticmethod
    def strip_punct(x):
        # punct = set(punctuation)
        # punct = punct.difference(['\'', '.', ',', '?', ';'])
        # punct_re = re.compile('[%s\n]' % re.escape(''.join(punct)))
        # return re.sub(punct_re, '', x)
        return re.sub(r"[^\w\s\.\?,]", "", x)

    @staticmethod
    def apply(x, func_names):
        for func_name in func_names:
            x = getattr(PostProcessors, func_name, lambda x: x)(x)
        return x


def replace_item(t: np.ndarray, src: int, dest: int):
    t[t == src] = dest
    return t


def evaluate(df, tokenizer):
    sys.path.append("/home/user/")
    from metrics.eval_metrics import Evaluator
    import datasets

    evaluator = Evaluator(tokenizer=tokenizer)
    ds = datasets.Dataset.from_pandas(df)
    metrics = evaluator.compute_metrics(ds, pred_col="predictions", gt_col="response")
    logger.info("Eval metrics: \n", metrics)
    return metrics


def dump_all(
    output_dir: str,
    gen_hparams: GenerationHparams,
    metric_hparams: MetricHparams,
    prediction_df: pd.DataFrame,
    metrics_df: pd.DataFrame = None,
):
    if "context" in prediction_df.columns:
        prediction_df.to_pickle(os.path.join(output_dir, "ctx_resp_preds.pkl"))
        prediction_df.to_csv(
            os.path.join(output_dir, "ctx_resp_preds.csv"), index=False
        )
        prediction_df = prediction_df.drop(columns=["context"])

    prediction_df.to_pickle(os.path.join(output_dir, "eval_predictions.pkl"))
    prediction_df.to_csv(os.path.join(output_dir, "eval_predictions.csv"), index=False)
    if metrics_df is not None:
        metrics_df.to_csv(os.path.join(output_dir, "eval_metrics.csv"), index=False)
    json.dump(
        gen_hparams.__dict__,
        open(os.path.join(output_dir, "gen_hparams.json"), "w"),
    )
    json.dump(
        metric_hparams.__dict__,
        open(os.path.join(output_dir, "metric_hparams.json"), "w"),
    )
    logger.info("Dumped predictions to ", output_dir)


def process_and_dump_predictions(
    args: TrainingArguments,
    gen_hparams: GenerationHparams,
    metric_hparams: MetricHparams,
    eval_dataset: Dataset,
    eval_preds: PredictionOutput,
    make_ctx=None,
):
    lengths = [
        len(eval_dataset[i, False]["input_ids"])
        for i in range(len(eval_preds.predictions))
    ]

    predictions = replace_item(
        eval_preds.predictions, -100, eval_dataset.tokenizer.eos_token_id
    )
    predictions = [x[lengths[i] :] for i, x in enumerate(predictions)]

    label_ids = replace_item(
        eval_preds.label_ids, -100, eval_dataset.tokenizer.eos_token_id
    )
    prediction_txt = eval_dataset.tokenizer.batch_decode(
        predictions, skip_special_tokens=True
    )
    label_txt = eval_dataset.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    prediction_txt_pp = [
        PostProcessors.apply(x, metric_hparams.post_process) for x in prediction_txt
    ]

    pred_df = pd.DataFrame(
        {
            "predictions": prediction_txt_pp,
            "predictions_original": prediction_txt,
            "response": label_txt,
        }
    )
    try:
        if make_ctx:
            pred_df["context"] = make_ctx(eval_dataset)
        
        # if "persona" in eval_dataset.data.column_names:
        #     pred_df["persona"] = eval_dataset.data["persona"]
        
        metrics_df = evaluate(pred_df, eval_dataset.tokenizer)
    except Exception as e:
        print("Error occured in evaluation", e)
        metrics_df = None
    finally:
        dump_all(
            output_dir=args.output_dir,
            gen_hparams=gen_hparams,
            metric_hparams=metric_hparams,
            prediction_df=pred_df,
            metrics_df=metrics_df,
        )

    return pred_df, metrics_df
