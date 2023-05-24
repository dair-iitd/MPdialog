from dataclasses import dataclass, field
from typing import Tuple

from transformers.hf_argparser import HfArgumentParser
from typing import List


@dataclass
class GenerationHparams:
    do_sample: bool = True
    num_beams: int = 1
    max_new_tokens: int = 50
    repetition_penalty: float = 1.2
    temperature: float = 0.05
    top_k: int = 50
    top_p: float = 0.95
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1


@dataclass
class MetricHparams:
    post_process: List[str] = field(default_factory=lambda: ["strip_punct"])


@dataclass
class TrainingHParams:
    eval_steps: int = 4000
    save_steps: int = 4000
    logging_steps: int = 1000
    warmup_steps: int = 500
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "linear"
    adam_epsilon: float = 1e-8
    weight_decay: int = 0
    save_total_limit: int = 5
    eval_accumulation_steps: int = 1
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 5
    # early_stopping_threshold: float = 0.1
    train_batch_size: int = 12
    eval_batch_size: int = 12
    prediction_loss_only: bool = False


@dataclass
class TrainingParams:
    output_dir: str
    do_train: bool
    do_eval: bool
    do_predict: bool
    do_freeze: bool = True
    n_visual_tokens: int = 1
    mapper_type: str = "linear"
    resume_from_checkpoint: bool = False
    num_train_epochs: int = 5
    run_name: str = "test"
    model_init_dir: str = "models/base"
    lm_init_dir: str = None
    visual_init_dir: str = None
    tokenizer_dir: str = "models/base/lm"
    # tokenizer_dir: str = "/home/user/PersonaGPT/personaGPT-comic-small-p/checkpoint/model"
    visual_preprocessor_dir: str = "models/base/visual"
    persona_path: str = None
    # persona_path: str = "/home/user/datasets/persona1.yaml"
    # persona_path: str = "/home/user/persona_out_all.yml"
    mapper_init_dir: str = None
    # data_dir: str = "data/coco"
    data_dir: str = "/home/user/comic-visual/multimodal_data_corrected/"
    data_source: str = "comic"  # "coco" or "comic"
    include_current_image: bool = True
    train_lm_only: bool = False
    use_double_eos_in_dialogues: bool = False
    split_for_validation: str = "validation"
    domain: str = "in-domain"


def get_args() -> Tuple[TrainingParams, TrainingHParams]:
    parser = HfArgumentParser(
        [TrainingParams, TrainingHParams, GenerationHparams, MetricHparams]
    )

    (
        params,
        hparams,
        gen_hparams,
        metric_hparams,
        unknown_args,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # Optionally, show a warning on unknown arguments.
    return params, hparams, gen_hparams, metric_hparams
