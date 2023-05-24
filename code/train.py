import sys
from typing import Tuple
import os
from transformers.trainer import Trainer
from model import Frozen, FrozenConfig, GPTLMHeadForGeneration
from transformers.training_args import TrainingArguments
from transformers import (
    EarlyStoppingCallback,
    GPT2Tokenizer,
    CLIPFeatureExtractor,
    CLIPProcessor,
    GPT2LMHeadModel,
)
import logging
import pandas as pd
import json
import numpy as np

from params import get_args
from data import CocoDataset, FrozenCollator, ComicDataset, GPTCollator
from predict_util import process_and_dump_predictions


logger = logging.getLogger(__name__)


if __name__ == "__main__":

    params, hparams, gen_hparams, metric_hparams = get_args()
    (
        params.lm_init_dir,
        params.visual_init_dir,
        params.mapper_init_dir,
    ) = Frozen._make_paths(
        params.model_init_dir,
        params.lm_init_dir,
        params.visual_init_dir,
        params.mapper_init_dir,
    )
    print("Params: ", params)
    print("Hparams:", hparams)
    print("GenHparams:", gen_hparams)
    print("MetricHparams:", gen_hparams)
    args = TrainingArguments(
        output_dir=params.output_dir,
        overwrite_output_dir=True,
        do_train=params.do_train,
        do_eval=params.do_eval,
        do_predict=params.do_predict,
        evaluation_strategy="steps",
        eval_accumulation_steps=hparams.eval_accumulation_steps,
        gradient_accumulation_steps=hparams.gradient_accumulation_steps,
        eval_steps=hparams.eval_steps,
        per_device_train_batch_size=hparams.train_batch_size,
        per_device_eval_batch_size=hparams.eval_batch_size,
        learning_rate=hparams.learning_rate,
        lr_scheduler_type=hparams.lr_scheduler_type,
        adam_epsilon=hparams.adam_epsilon,
        weight_decay=hparams.weight_decay,
        num_train_epochs=params.num_train_epochs,
        warmup_steps=hparams.warmup_steps,
        log_level="info",
        logging_strategy="steps",
        logging_steps=hparams.logging_steps,
        save_strategy="steps",
        save_steps=hparams.save_steps,
        save_total_limit=hparams.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
        if hparams.prediction_loss_only
        else "eval_ppl",
        greater_is_better=False,
        # group_by_length=False,
        run_name=params.run_name,
        report_to="wandb",
        full_determinism=True,
        prediction_loss_only=hparams.prediction_loss_only,
        # auto_find_batch_size=True,
    )

    tokenizer = GPT2Tokenizer.from_pretrained(params.tokenizer_dir)
    transform = CLIPProcessor.from_pretrained(params.visual_preprocessor_dir)
    # transform = CLIPFeatureExtractor()
    if params.data_source == "coco":
        train_dataset = CocoDataset(
            root_dir=os.path.join(params.data_dir, "train2017"),
            ann_file=os.path.join(
                params.data_dir, "annotations/captions_train2017.json"
            ),
            preprocessor_dir=params.visual_preprocessor_dir,
            tokenizer_dir=params.tokenizer_dir,
            n_visual_tokens=params.n_visual_tokens if not params.train_lm_only else 0,
        )

        eval_dataset = CocoDataset(
            root_dir=os.path.join(params.data_dir, "val2017"),
            ann_file=os.path.join(params.data_dir, "annotations/captions_val2017.json"),
            preprocessor_dir=params.visual_preprocessor_dir,
            tokenizer_dir=params.tokenizer_dir,
            n_visual_tokens=params.n_visual_tokens if not params.train_lm_only else 0,
            load_captions_as_inputs=not args.do_predict,
        )
    elif params.data_source == "comic":
        try:
            train_dataset = ComicDataset(
                base=params.data_dir,
                split="train",
                tokenizer=tokenizer,
                persona_path=params.persona_path,
                n_visual_tokens=params.n_visual_tokens
                if not params.train_lm_only
                else 0,
                transform=transform,
                include_current_image=params.include_current_image,
                use_double_eos_in_dialogues=params.use_double_eos_in_dialogues,
                domain=params.domain,
            )
        except:
            print("Could not load train dataset")
            train_dataset = []

        eval_dataset = ComicDataset(
            base=params.data_dir,
            split=params.split_for_validation,
            tokenizer=tokenizer,
            persona_path=params.persona_path,
            n_visual_tokens=params.n_visual_tokens if not params.train_lm_only else 0,
            transform=transform,
            load_captions_as_inputs=not args.do_predict,
            include_current_image=params.include_current_image,
            use_double_eos_in_dialogues=params.use_double_eos_in_dialogues,
            domain=params.domain,
        )

    logger.info("Train size: ", len(train_dataset))
    logger.info("Eval size: ", len(eval_dataset))

    collator_cls = GPTCollator if params.train_lm_only else FrozenCollator
    data_collator = collator_cls(
        gen_hparams=None if not params.do_predict else gen_hparams
    )

    def model_init():

        if params.train_lm_only:
            model = GPTLMHeadForGeneration.from_pretrained(params.lm_init_dir)
            print("Resizing model to tokenizer vocab size ", len(tokenizer))
            model.resize_token_embeddings(len(tokenizer))
            return model

        if not params.do_predict:

            model = Frozen.from_pretrained(
                dir_path=params.model_init_dir,
                lm_path=params.lm_init_dir,
                visual_path=params.visual_init_dir,
                mapping_model_dir=params.mapper_init_dir,
                config=FrozenConfig(
                    n_visual_tokens=params.n_visual_tokens,
                    mapper_type=params.mapper_type,
                ),
            )
            model.lm.resize_token_embeddings(len(tokenizer))
            return model.freeze_lm() if params.do_freeze else model
        model = Frozen.from_pretrained(dir_path=params.model_init_dir)
        model.lm.resize_token_embeddings(len(tokenizer))
        return model

    trainer = Trainer(
        args=args,
        data_collator=data_collator.collate,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_init=model_init,
        compute_metrics=Frozen.compute_metrics if not params.do_predict else None,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=hparams.early_stopping_patience,
                early_stopping_threshold=0.001
                if args.metric_for_best_model == "eval_loss"
                else 0.1,
            )
        ],
    )
    if args.do_train:
        logger.info("Starting training")

        trainer.train(resume_from_checkpoint=params.resume_from_checkpoint)
        trainer.save_model(os.path.join(args.output_dir, "best_model"))
    if args.do_predict:
        logger.info("Starting predictions on eval set")
        eval_preds = trainer.predict(eval_dataset)
        
        def _coco_context(eval_dataset):
            input_ids = [eval_dataset[i]["input_ids"] for i in range(len(eval_dataset))]
            input_txt = eval_dataset.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True
            )
            return input_txt

        def _comic_context(eval_dataset):

            input_ids = [
                eval_dataset[i, False]["input_ids"] for i in range(len(eval_dataset))
            ]
            start_token_id = eval_dataset.tokenizer.convert_tokens_to_ids(
                ["<|start|>"]
            )[0]

            def get_index(x):
                try:
                    return x.index(start_token_id)
                except ValueError:
                    return 0

            input_ids = [x[get_index(x) :] for x in input_ids]

            input_txt = eval_dataset.tokenizer.batch_decode(
                input_ids, skip_special_tokens=False
            )
            return input_txt

        make_ctx = _coco_context if params.data_source == "coco" else _comic_context

        process_and_dump_predictions(
            args,
            gen_hparams,
            metric_hparams,
            eval_dataset,
            eval_preds,
            make_ctx=make_ctx,
        )

    elif args.do_eval:
        logger.info("Starting evaluation")
        comic_groups = eval_dataset.data.to_pandas().groupby(["comic"]).groups
        metric_dict = {}
        lengths = []
        losses = []
        for k, v in comic_groups.items():
            print(k, len(v))
            lengths.append(len(v))
            eval_dataset.keep(v.tolist())
            metrics = trainer.evaluate(eval_dataset)
            metric_dict[k] = metrics
            print("metrics: ", metrics)
            losses.append(metrics["eval_loss"])
            trainer.save_metrics(k, metrics)

        lengths = np.array(lengths)
        losses = np.array(losses)

        weighted_loss = np.sum(losses * lengths) / np.sum(lengths)
        overall_metrics = {
            "eval_loss": weighted_loss,
        }

        trainer.save_metrics(params.split_for_validation, overall_metrics)
