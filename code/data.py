from typing import List
import torchvision
from torch.utils.data import Dataset
from transformers import CLIPProcessor, GPT2Tokenizer
import torch
import numpy as np
from PIL import Image
import os
import os.path
import datasets

datasets.set_caching_enabled(False)
from utils import *


class FrozenCollator:
    def __init__(self, gen_hparams=None) -> None:
        self.gen_hparams = gen_hparams

    def collate(self, batch):
        collated_batch = FrozenCollator.collate_fn(batch)
        if self.gen_hparams is not None:
            collated_batch["gen_hparams"] = self.gen_hparams
        return collated_batch

    @staticmethod
    def collate_fn(batch):
        max_input_len = max(len(item["input_ids"]) for item in batch)
        max_labels_len = max(len(item["labels"]) for item in batch)

        input_ids = torch.vstack(
            [
                torch.cat(
                    [
                        torch.Tensor(item["input_ids"]),
                        torch.zeros(max_input_len - len(item["input_ids"])),
                    ]
                )
                for item in batch
            ]
        ).long()

        image_mask = torch.vstack(
            [
                torch.cat(
                    [
                        torch.Tensor(item["image_mask"]),
                        torch.zeros(max_input_len - len(item["input_ids"])),
                    ]
                )
                for item in batch
            ]
        ).bool()

        attention_mask = torch.vstack(
            [
                torch.cat(
                    [
                        torch.Tensor(item["attention_mask"]),
                        torch.zeros(max_input_len - len(item["input_ids"])),
                    ]
                )
                for item in batch
            ]
        ).bool()
        labels = torch.vstack(
            [
                torch.cat(
                    [
                        torch.Tensor(item["labels"]),
                        -100 * torch.ones(max_labels_len - len(item["labels"])),
                    ]
                )
                for item in batch
            ]
        ).long()
        try:
            images = torch.vstack(
                [
                    torch.Tensor(item["images"])
                    for item in batch
                    if len(item["images"]) > 0
                ]
            )
        except:
            images = torch.Tensor([])

        collated_batch = {
            "input_ids": input_ids,
            "images": images,
            "image_mask": image_mask,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        # for k, v in collated_batch.items():
        #     print(k, v.shape)
        return collated_batch


class GPTCollator:
    def __init__(self, gen_hparams=None) -> None:
        self.gen_hparams = gen_hparams

    def collate(self, batch):
        collated_batch = GPTCollator.collate_fn(batch)
        if self.gen_hparams is not None:
            collated_batch["gen_hparams"] = self.gen_hparams
        return collated_batch

    @staticmethod
    def collate_fn(batch):
        max_input_len = max(len(item["input_ids"]) for item in batch)
        max_labels_len = max(len(item["labels"]) for item in batch)

        input_ids = torch.vstack(
            [
                torch.cat(
                    [
                        torch.Tensor(item["input_ids"]),
                        torch.zeros(max_input_len - len(item["input_ids"])),
                    ]
                )
                for item in batch
            ]
        ).long()

        # image_mask = torch.vstack(
        #     [
        #         torch.cat(
        #             [
        #                 torch.Tensor(item["image_mask"]),
        #                 torch.zeros(max_input_len - len(item["input_ids"])),
        #             ]
        #         )
        #         for item in batch
        #     ]
        # ).bool()

        attention_mask = torch.vstack(
            [
                torch.cat(
                    [
                        torch.Tensor(item["attention_mask"]),
                        torch.zeros(max_input_len - len(item["input_ids"])),
                    ]
                )
                for item in batch
            ]
        ).bool()
        labels = torch.vstack(
            [
                torch.cat(
                    [
                        torch.Tensor(item["labels"]),
                        -100 * torch.ones(max_labels_len - len(item["labels"])),
                    ]
                )
                for item in batch
            ]
        ).long()
        # try:
        #     images = torch.vstack(
        #         [
        #             torch.Tensor(item["images"])
        #             for item in batch
        #             if len(item["images"]) > 0
        #         ]
        #     )
        # except:
        #     images = torch.Tensor([])

        collated_batch = {
            "input_ids": input_ids,
            # "images": images,
            # "image_mask": image_mask,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        # for k, v in collated_batch.items():
        #     print(k, v.shape)
        return collated_batch


# # TODO: <modify>
# class BertCollator:
#     def __init__(self, gen_hparams=None) -> None:
#         self.gen_hparams = gen_hparams

#     def collate(self, batch):
#         collated_batch = GPTCollator.collate_fn(batch)
#         if self.gen_hparams is not None:
#             collated_batch["gen_hparams"] = self.gen_hparams
#         return collated_batch

#     @staticmethod
#     def collate_fn(batch):
#         max_input_len = max(len(item["input_ids"]) for item in batch)
#         max_labels_len = max(len(item["labels"]) for item in batch)

#         input_ids = torch.vstack(
#             [
#                 torch.cat(
#                     [
#                         torch.Tensor(item["input_ids"]),
#                         torch.zeros(max_input_len - len(item["input_ids"])),
#                     ]
#                 )
#                 for item in batch
#             ]
#         ).long()

#         attention_mask = torch.vstack(
#             [
#                 torch.cat(
#                     [
#                         torch.Tensor(item["attention_mask"]),
#                         torch.zeros(max_input_len - len(item["input_ids"])),
#                     ]
#                 )
#                 for item in batch
#             ]
#         ).bool()
#         labels = torch.vstack(
#             [
#                 torch.cat(
#                     [
#                         torch.Tensor(item["labels"]),
#                         -100 * torch.ones(max_labels_len - len(item["labels"])),
#                     ]
#                 )
#                 for item in batch
#             ]
#         ).long()

#         collated_batch = {
#             "input_ids": input_ids,
#             "labels": labels,
#             "attention_mask": attention_mask,
#         }

#         return collated_batch


class CocoDataset(Dataset):
    def __init__(
        self,
        root_dir,
        ann_file,
        preprocessor_dir,
        tokenizer_dir,
        n_visual_tokens,
        load_captions_as_inputs=True,
    ) -> None:
        super().__init__()
        coco_cap = torchvision.datasets.CocoCaptions(root=root_dir, annFile=ann_file)
        self.coco_cap = coco_cap
        self.n_visual_tokens = n_visual_tokens
        self.preprocessor = CLIPProcessor.from_pretrained(preprocessor_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_dir)
        self.load_captions_as_inputs = load_captions_as_inputs

    def __len__(self):
        # return 64
        return len(self.coco_cap)

    def __getitem__(self, index):
        # print(index)
        if index >= self.__len__():
            print("Loaded all data")
            raise StopIteration
        img, target = self.coco_cap[index]
        img_input = self.preprocessor(images=[img], return_tensors="pt")["pixel_values"]
        caption = target[0]

        input_ids = (
            [self.tokenizer.unk_token_id] * self.n_visual_tokens
            + self.tokenizer.encode(caption)
            + [self.tokenizer.eos_token_id]
        )

        input_ids = torch.Tensor(input_ids)
        labels = input_ids.clone()

        if not self.load_captions_as_inputs:
            input_ids = input_ids[: self.n_visual_tokens]

        labels[: self.n_visual_tokens] = -100

        attention_mask = torch.ones_like(input_ids).long()
        image_mask = torch.zeros_like(input_ids).long()
        image_mask[: self.n_visual_tokens] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": img_input,
            "image_mask": image_mask,
        }


class ComicDataset(Dataset):
    def __init__(
        self,
        base,
        split,
        tokenizer,
        persona_path=None,
        n_visual_tokens=1,
        masking=True,
        transform=None,
        domain="in-domain",
        load_captions_as_inputs=True,
        include_current_image=True,
        use_double_eos_in_dialogues=False,
    ):
        # self.max_len = max_len if max_len is not None else MAX_LEN
        self.base = base
        self.n_visual_tokens = n_visual_tokens
        self.include_current_image = include_current_image
        # self.vocab = vocab
        self.personas = (
            load_persona(persona_path, eos_token=tokenizer.eos_token)
            if persona_path
            else None
        )
        self.load_captions_as_inputs = load_captions_as_inputs
        self.split = split
        self.domain = domain
        self.tokenizer = tokenizer
        self.transform = transform
        self.data = fetch_dataset(os.path.join(base, "text_data", domain), split)
        self.data = self.data.map(
            get_ctx_resp,
            batched=True,
            remove_columns=self.data.column_names,
            load_from_cache_file=False,
            fn_kwargs={
                "personas": self.personas,
                "tokenizer": self.tokenizer,
                "max_history": 5,
                "n_visual_tokens": n_visual_tokens,
                "include_current_image": include_current_image,
                "use_double_eos_in_dialogues": use_double_eos_in_dialogues,
            },
        )
        self.visual_path = os.path.join(base, "vision_data", domain)
        # bboxes = fetch_dataset(os.path.join(base, "vision_data", domain), split).map(
        bboxes = datasets.load_from_disk(os.path.join(base, "ocr_data"))
        bboxes = bboxes.map(lambda x: {"comic_name": x["comic_name"].split("_")[0]})
        self.bboxes = {}
        for i in range(len(bboxes)):
            if bboxes[i]["comic_name"] not in self.bboxes:
                self.bboxes[bboxes[i]["comic_name"]] = {}
            self.bboxes[bboxes[i]["comic_name"]][bboxes[i]["strip_sequence"]] = bboxes[
                i
            ]["ocr_bboxes"]
            # self.bboxes.filter(lambda x: (comic_name == x['comic_name']) and (x['strip_sequence']==strip_sequence))

        self.masking = masking
        self.keep_idx_list = None

    def keep(self, idx_list: List[int]):
        assert max(idx_list) < len(self.data)

        self.keep_idx_list = idx_list

    def __getitem__(self, item):
        # vocab = self.vocab
        if isinstance(item, int):
            idx = item
            do_load_image = True
        elif isinstance(item, tuple):
            idx, do_load_image = item

        x = (
            self.data[idx]
            if self.keep_idx_list is None
            else self.data[self.keep_idx_list[idx]]
        )
        labels = []
        input_ids = []
        image_mask = []
        # if self.load_captions_as_inputs:
        for strip in (x["context"]).split(IMAGE_TOKEN):
            tok = self.tokenizer.encode(strip)
            input_ids += tok + [self.tokenizer.eos_token_id] * self.n_visual_tokens
            # labels += tok + [-100] * self.n_visual_tokens
            labels += [-100] * (len(tok) + self.n_visual_tokens)
            image_mask += [False] * len(tok) + [True] * self.n_visual_tokens
        if self.n_visual_tokens > 0:
            input_ids = input_ids[: -self.n_visual_tokens]
            labels = labels[: -self.n_visual_tokens]
            image_mask = image_mask[: -self.n_visual_tokens]

        response_ids = self.tokenizer.encode(x["response"])
        labels += response_ids
        if self.load_captions_as_inputs:
            input_ids += response_ids
            image_mask += [False] * len(response_ids)

        try:
            images = (
                self.get_images(x) if do_load_image and self.n_visual_tokens else None
            )
        except Exception as e:
            print(idx)
            print(x)
            raise e
        # assert len(images) * self.n_visual_tokens == sum(
        #     image_mask
        # ), f"Number of images {len(images)} does not match number of masks ids {sum(image_mask)} for no of tokens: {self.n_visual_tokens},{image_mask}"
        return {
            "input_ids": input_ids,
            "images": images,
            "image_mask": image_mask,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "path": x["comic"] + " " + x["sequence"],
            "len": len(input_ids),
        }

    def get_bbox(self, comic_name, strip_sequence, idx):
        comic_name = comic_name.split("_")[0]
        result = self.bboxes.get(comic_name, {}).get(strip_sequence, [])
        # self.bboxes.filter(lambda x: (comic_name == x['comic_name']) and (x['strip_sequence']==strip_sequence))
        # assert len(result)<=1, f'Multiple entries found for comic_name: {comic_name} and strip_sequence: {strip_sequence}'
        if result:
            for x in result[idx]:
                x[:4] = [round(y) for y in x[:4]]
                if x[0] == x[2] or x[1] == x[3]:
                    continue
                yield min(x[0], x[2]), min(x[1], x[3]), max(x[0], x[2]), max(x[1], x[3])
        else:
            pass

            # print(
            #     "No bbox found for comic_name: ",
            #     comic_name,
            #     " and strip_sequence: ",
            #     strip_sequence,
            # )
            # import pdb
            # pdb.set_trace()

    def get_mask(self, bbox, mask_type="random"):
        x0, y0, x1, y1 = bbox
        if mask_type == "random":
            return Image.fromarray(
                np.random.randint(0, 256, (abs(y1 - y0), abs(x1 - x0), 3)), "RGB"
            )

    def get_images(self, x):
        for data_split in ["train", "test", "validation", "all"]:
            path = os.path.join(
                self.visual_path,
                data_split,
                x["comic"].split("_")[0],
                x["sequence"].split(" ")[-1],
            )
            if os.path.exists(path):
                break
        imgs = []
        for id_ in x["image_ids"]:
            img = Image.open(os.path.join(path, f"{id_}.jpg")).convert("RGB")
            # img_w, img_h = img.size
            # The random transform
            if self.masking:
                for bbox in self.get_bbox(x["comic"], x["sequence"], id_):

                    # m_w, m_h = mask.size
                    # img.paste(mask, ((img_w - m_w) // 2, (img_h - m_h) // 2))
                    try:
                        mask = self.get_mask(bbox)
                        img.paste(mask, bbox)
                    except Exception as e:
                        print(img.size, mask.size, bbox)
                        print(img)
                        print(mask)
                        print(bbox)
                        raise e
            imgs.append(img)

        if self.transform and len(imgs):
            imgs = self.transform(images=imgs, return_tensors="pt")["pixel_values"]
            # imgs = [self.transform(image)['pixel_values'] for image in imgs]
        # imgs = torch.Tensor(np.array(imgs))
        assert len(imgs) == len(
            x["image_ids"]
        ), f'Number of images {len(imgs)} does not match number of image ids {len(x["image_ids"])}'
        return imgs

    def __len__(self):
        return len(self.data) if self.keep_idx_list is None else len(self.keep_idx_list)
        # return 24

    # def collate(self,batch):


class NLIDataset(Dataset):
    def __init__(self, pre, hyp):
        self.pre = pre
        self.hyp = hyp

    def __getitem__(self, idx):
        pre = {
            key: torch.tensor(val[idx]).to(self.device) for key, val in self.pre.items()
        }
        hyp = {
            key: torch.tensor(val[idx]).to(self.device) for key, val in self.hyp.items()
        }
        return {"pre": pre, "hyp": hyp}

    def __len__(self):
        return len(self.pre["input_ids"])
