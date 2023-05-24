import contextlib
import os
import pickle
import subprocess
import sys
import tempfile
import json

ROOT_DIR = "/home/user"


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def c_score(file_name):
    sys.path.append("PAML/")
    from utils.load_bert import bert_model

    bert = bert_model()

    preds, personas = pickle.load(open(file_name, "rb"))
    score = bert.predict_label(turn=preds, personas_items=personas)
    pickle.dump(score, open(file_name + ".c_score.pkl", "wb"))


def bert_score(file_name):
    from evaluate import load

    references, predictions = pickle.load(open(file_name, "rb"))

    bertscorer = load("evaluate-metric/bertscore")
    results = bertscorer.compute(
        predictions=predictions, references=references, lang="en"
    )
    ret = sum(results["f1"]) / len(results["f1"])
    pickle.dump(ret, open(file_name + ".bertscore.pkl", "wb"))


def bleurt_score(
    file_name,
    bleurt_checkpoint=f"{ROOT_DIR}/user/metrics/bleurt/bleurt/BLEURT-20",
):
    from bleurt import score

    references, predictions = pickle.load(open(file_name, "rb"))

    scorer = score.BleurtScorer(bleurt_checkpoint)
    scores = scorer.score(references=references, candidates=predictions)
    ret = sum(scores) / len(scores)
    pickle.dump(ret, open(file_name + ".bleurtscore.pkl", "wb"))


def maude(file_name):
    cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    with tempfile.TemporaryDirectory() as d:

        # FIXME
        with pushd(f"{ROOT_DIR}/user/metrics/maude/online_dialog_eval"):
            # PYTHON = f"{ROOT_DIR}/bin/anaconda3/envs/metrics_n/bin/python"
            PYTHON = f"/home/user/anaconda3/envs/metrics_n/bin/python"
            MODEL_SAVE_DIR = "full_acl_runs/"
            DATA_NAME = "convai2"
            FINE_TUNE_MODEL = "convai2_data/distilbert_lm"
            TRAIN_MODE = "nce"
            VERSION = "20488119"
            MODEL_ID = "na_all"
            subprocess.run(
                f"bash -c 'CUDA_VISIBLE_DEVICES={cuda_device} {PYTHON} codes/inference.py --id {MODEL_ID} --model_save_dir {MODEL_SAVE_DIR} --model_version {VERSION} --train_mode {TRAIN_MODE} --corrupt_pre {file_name} --test_column \"predictions\" --results_file {d}/maude.jsonl'",
                # env=os.environ,
                shell=True,
            )
            maude_data = json.load(open(f"{d}/maude.jsonl", "r"))
            ret = maude_data["score_mean"]
            pickle.dump(ret, open(file_name + ".maude.pkl", "wb"))


if __name__ == "__main__":
    scorer, file_name = sys.argv[1], sys.argv[2]
    scorer2fn = {
        "bertscore": bert_score,
        "bleurtscore": bleurt_score,
        "maude": maude,
        "c_score": c_score,
    }
    scorer2fn[scorer](file_name)
