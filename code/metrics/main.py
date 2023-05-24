import argparse
from eval_metrics import Evaluator
import datasets
import pandas as pd
from transformers import AutoTokenizer, GPT2Tokenizer


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-df", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--no-neural", action="store_true")
    parser.add_argument("--context-col", type=str, default="context")
    parser.add_argument("--persona-col", type=str, default="persona")
    parser.add_argument("--pred-col", type=str, default="predictions")
    parser.add_argument("--gt-col", type=str, default="response")
    parser.add_argument("--tokenizer-dir", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    except:
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_dir)

    eval = Evaluator(tokenizer)
    df = pd.read_pickle(args.input_df)
    cols = [args.pred_col, args.gt_col]
    if args.context_col in df.columns:
        cols.append(args.context_col)
    if args.persona_col in df.columns:
        cols.append(args.persona_col)
    dataset = datasets.Dataset.from_pandas(df[cols])

    metrics = eval.compute_metrics(
        dataset,
        pred_col=args.pred_col,
        gt_col=args.gt_col,
        persona_col=args.persona_col,
        context_col=args.context_col,
        compute_neural=not args.no_neural,
    )
    print(metrics)

    metrics.to_csv(args.output)
