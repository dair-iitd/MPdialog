import datasets
from transformers import AutoTokenizer

from multiset import Multiset
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.util import ngrams
import numpy as np
import pandas as pd
import nltk
import re
from tqdm import tqdm
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import CiderScorer
import tempfile
import contextlib
import os
import subprocess
import json
import pickle
from itertools import product


# from util import clean_predictions, get_dataset, get_edge_path
def flatten(ls):
    return [y for x in ls for y in x]


class Metrics:
    def __init__(self):
        self.prec = []
        self.rec = []
        self.f1 = []
        self.bleu4_scores = []
        self.meteor_scores = []
        self.rouge_scores = []
        self.cider_scorer = CiderScorer()
        # keys = n, val = List[n-grams]
        self.ngrams = {}

    def mean(self):
        ret_metrics = Metrics()
        for k, v in self.__dict__.items():
            if isinstance(v, list):
                setattr(ret_metrics, k, np.mean(v))
            elif isinstance(v, CiderScorer):
                score, _ = v.compute_score()
                setattr(ret_metrics, "cider", score)
                delattr(ret_metrics, "cider_scorer")
            else:
                delattr(ret_metrics, k)

        for n, ngrams_list in self.ngrams.items():
            s = set()
            l = 0
            # import pickle
            # open(f'{CACHE_DIR}/ngrams_{n+1}.bin','wb').write(pickle.dumps(ngrams_list))

            for ngrams in tqdm(ngrams_list, desc=f"Distinct-{n+1}"):
                l += len(ngrams)
                s = s.union(set(ngrams))

            setattr(ret_metrics, f"distinct_{n+1}", len(s) / l)
            # setattr(ret_metrics, f'distinct_{n+1}', len(s)/self.len_sentences)

        return ret_metrics


class Evaluator:
    def __init__(self, tokenizer, cc=SmoothingFunction().method3, n_dist=3):
        self.tokenizer = tokenizer
        self.cc = cc
        self.n_dist = n_dist
        self.metrics = Metrics()
        self.metrics.ngrams = {i: [] for i in range(self.n_dist)}

        self.rouge_scorer = Rouge()

    def eval_sample(
        self, pred_set: Multiset, truth_set: Multiset, prediction_txt, label_txt
    ):
        is_empty = len(pred_set) == 0

        intersection, union = len(truth_set.intersection(pred_set)), len(
            truth_set.union(pred_set)
        )
        if len(truth_set) == 0:
            return
        prec = intersection / len(pred_set) if not is_empty else 0
        rec = intersection / len(truth_set)
        self.metrics.prec.append(prec)
        self.metrics.rec.append(rec)
        self.metrics.f1.append((2 * prec * rec) / (prec + rec + 1e-3))

        line = re.sub(r"\s+", " ", prediction_txt.strip())
        sent = nltk.word_tokenize(line)

        for i in range(self.n_dist):

            computed_ngrams = list(ngrams(sent, i + 1))
            self.metrics.ngrams[i].append(computed_ngrams)

        bleu_score = (
            sentence_bleu(
                [label_txt.split()],
                prediction_txt.split(),
                smoothing_function=self.cc,
                auto_reweigh=True,
                weights=(0, 0, 0, 1),
            )
            if not is_empty
            else 0
        )
        _meteor_score = meteor_score([label_txt.split()], prediction_txt.split())

        self.metrics.cider_scorer += (prediction_txt, [label_txt])

        self.metrics.rouge_scores.append(
            self.rouge_scorer.calc_score(candidate=[prediction_txt], refs=[label_txt])
        )
        self.metrics.meteor_scores.append(_meteor_score)
        self.metrics.bleu4_scores.append(bleu_score)

    def eval_samples(self, samples, pred_col="prediction", gt_col="response"):
        """
        samples is a dataset batch of `pred_col`, `gt_col`
        """
        special_tokens = set(
            [
                getattr(self.tokenizer, k)
                for k in self.tokenizer.__dir__()
                if k.endswith("_id") and isinstance(getattr(self.tokenizer, k), int)
            ]
        )
        for pred, response in zip(samples[pred_col], samples[gt_col]):
            pred_ids = Multiset(self.tokenizer.encode(pred))
            resp_ids = Multiset(self.tokenizer.encode(response))
            for k in special_tokens:
                pred_ids.discard(k)
                resp_ids.discard(k)

            self.eval_sample(
                pred_set=pred_ids,
                truth_set=resp_ids,
                prediction_txt=pred,
                label_txt=response,
            )

    def compute_neural_metrics(
        self, ds, pred_col, gt_col, context_col=None, persona_col=None
    ):
        predictions = ds[pred_col]
        references = ds[gt_col]
        f1 = tempfile.NamedTemporaryFile()
        f2 = tempfile.NamedTemporaryFile()
        f3 = tempfile.NamedTemporaryFile()
        pickle.dump((references, predictions), f1)
        f1.flush()

        ds.to_pandas().to_pickle(f2)
        f2.flush()

        neural_metrics_to_compute = ["bleurtscore", "bertscore"]
        input_files = [f1.name, f1.name]
        scores = []

        if context_col in ds.column_names:
            neural_metrics_to_compute += ["maude"]
            input_files += [f2.name]
        # import pdb

        # pdb.set_trace()
        if persona_col in ds.column_names:
            neural_metrics_to_compute += ["c_score"]
            pred_persona = (
                ds.to_pandas()
                .apply(lambda x: product([x[pred_col]], x[persona_col]), axis=1)
                .tolist()
            )
            pred_persona = flatten(pred_persona)
            preds, personas = list(zip(*pred_persona))  # unzip
            pickle.dump((preds, personas), f3)
            f3.flush()
            input_files += [f3.name]

        neural_metrics_to_compute = list(reversed(neural_metrics_to_compute))
        input_files = list(reversed(input_files))

        print("Will compute: ", neural_metrics_to_compute)
        neural_metric_script = os.path.join(
            os.path.dirname(__file__), "neural_metrics.py"
        )
        for neural_metric, input_file in zip(neural_metrics_to_compute, input_files):
            # for neural_metric, input_file in reversed(
            # list(zip(neural_metrics_to_compute, input_files))
            # ):

            # if neural_metric != "c_score":
            # continue

            print(f"[*] Computing {neural_metric}")
            subprocess.run(
                f"bash -c 'python {neural_metric_script} {neural_metric} {input_file}'",
                shell=True,
            )
            scores.append(pickle.load(open(f"{input_file}.{neural_metric}.pkl", "rb")))
            os.remove(f"{input_file}.{neural_metric}.pkl")

        f1.close()
        f2.close()
        f3.close()
        return dict(zip(neural_metrics_to_compute, scores))

    def compute_metrics(
        self,
        ds,
        pred_col,
        gt_col,
        context_col=None,
        persona_col=None,
        batch_size=100,
        compute_neural=True,
    ):
        """
        dataset with keys `pred_col`, `gt_col`
        """
        if compute_neural:
            neural_metrics = self.compute_neural_metrics(
                ds,
                pred_col=pred_col,
                gt_col=gt_col,
                persona_col=persona_col,
                context_col=context_col,
            )
        else:
            neural_metrics = {}

        print("neural metrics: ", neural_metrics)
        ds.map(
            self.eval_samples,
            batched=True,
            batch_size=batch_size,
            remove_columns=ds.column_names,
            fn_kwargs={"pred_col": pred_col, "gt_col": gt_col},
        )
        metric_mean = self.metrics.mean()
        metric_df = pd.DataFrame(metric_mean.__dict__, index=[0])

        for k, v in neural_metrics.items():
            metric_df[k] = [v]
        return metric_df
