import datasets
import os
import yaml
from collections import defaultdict
import numpy as np

IMAGE_TOKEN = "<|img|> "


def fetch_dataset(dir, split=None):
    if split is None:
        return datasets.DatasetDict(
            {
                split: datasets.load_from_disk(os.path.join(dir, split))
                for split in os.listdir(dir)
                if "." not in split and os.path.isdir(os.path.join(dir, split))
            }
        )
    else:
        return datasets.load_from_disk(
            os.path.join(dir, split),
        )


def make_input_labels(examples, tokenizer=None, args=None):
    input_ids, labels = [], []
    for resp, ctx_ids in zip(examples["response"], examples["input_ids"]):
        resp = resp + tokenizer.eos_token + tokenizer.eos_token
        resp_ids = tokenizer.encode(resp)
        input_ids.append(ctx_ids)
        # labels.append([-100 for _ in range(len(ctx_ids))] + resp_ids)
        labels.append(resp_ids)
    return {"input_ids": input_ids, "labels": labels}


def get_ctx_resp(
    examples,
    personas=None,
    tokenizer=None,
    last_empty=False,
    args=None,
    max_history=-1,
    include_current_image=True,
    n_visual_tokens=1,
    use_double_eos_in_dialogues=False,
):

    resp = []
    context = []
    image_panels = []
    comic_name_l = []
    strip_sequence_l = []
    speakers_l = []
    personas_l = []
    eos_token = (
        tokenizer.eos_token + tokenizer.eos_token
        if use_double_eos_in_dialogues
        else tokenizer.eos_token
    )
    for dialogues, speakers, comic_name, p_ids, strip_sequence in zip(
        examples["dialogues"],
        examples["speakers"],
        examples["comic_name"],
        examples["panel_ids"],
        examples["strip_sequence"],
    ):
        if len([x for x in p_ids if x < 0]) > 0:
            p_ids = []
        prev_ctx = []
        prev_images = []
        # print(examples['comic_name'],comic,comic.split('\_'))
        comic = comic_name.split("_")[0]
        # for i in range(min(len(dialogues), len(p_ids))):
        for i in range(len(dialogues)):
            image_ids = []
            this_img = (
                p_ids[i] if i < len(p_ids) else p_ids[-1] if len(p_ids) > 0 else None
            )
            response = dialogues[i]
            speaker = speakers[i]
            persona_facts = personas[comic][speaker] if personas else ""
            ctx = (
                join_personas(persona_facts, eos_token=tokenizer.eos_token)
                if personas
                else ""
            )
            for dialouge, image_id in zip(prev_ctx, prev_images):

                if (
                    image_id not in image_ids
                    and image_id is not None
                    and (image_id != this_img or include_current_image)
                ):
                    ctx += " " + IMAGE_TOKEN + " "
                    image_ids.append(image_id)

                ctx += dialouge + " " + eos_token + " "
            if (
                include_current_image
                and this_img not in image_ids
                and this_img is not None
            ):
                ctx += " " + IMAGE_TOKEN + " "
                image_ids.append(this_img)

            prev_ctx.append(response)
            prev_images.append(this_img)
            if max_history > 0 and len(prev_ctx) > max_history:
                prev_ctx.pop(0)
                prev_images.pop(0)
            if i != 0:
                context.append(ctx)
                resp.append(response + " " + eos_token)
                speakers_l.append(speaker)
                personas_l.append(np.array(persona_facts, dtype=object))
                image_panels.append(image_ids)
                comic_name_l.append(comic_name)
                strip_sequence_l.append(strip_sequence)

    # print(type(context))
    ret_dict = {
        "context": context,
        "response": resp,
        "image_ids": image_panels,
        "comic": comic_name_l,
        "sequence": strip_sequence_l,
        "speaker": speakers_l,
    }
    if personas:
        ret_dict.update({"persona": personas_l})

    return ret_dict



def join_personas(personas, eos_token):
    return (
        "<|p|>"
        + ("".join(p_i + eos_token for p_i in personas) if personas else "")
        + "<|sep|>"
        + "<|start|>"
    )


def load_persona(path, eos_token):
    with open(path, "r+") as f:
        data = yaml.safe_load_all(f)
        data = list(data)
        dataf = defaultdict(lambda: defaultdict(lambda: "<|start|>"))
        for idx, elem in enumerate(data):
            if not elem:
                continue
            key = elem["comic"]
            del elem["comic"]
            dataf[key].update(elem)
        return dataf

        # dataf = {
        #     comic: defaultdict(
        #         lambda: "<|start|>",
        #         {
        #             char: "<|p|>"
        #             + ("".join(p_i + eos_token for p_i in p) if p else "")
        #             + "<|sep|>"
        #             + "<|start|>"
        #             for char, p in personas_comic.items()
        #         },
        #     )
        #     for comic, personas_comic in dataf.items()
        # }

    return dataf
