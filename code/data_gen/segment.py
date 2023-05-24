import torch
import torchvision
import os
import PIL as P
import random
import easyocr
import torchvision.transforms as T
import numpy as np
import datasets
import pandas as pd

from tqdm import tqdm
from pprint import pprint
from fuzzywuzzy import fuzz
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.transforms.functional import to_pil_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_img_path(comic_strip, base_dir="data"):
    return comic_strip["image_path"]


def build_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load(model_cls_dict,out_dir):
    model_dict = {}
    for k, v in model_cls_dict.items():
        m = v.load_state_dict(torch.load(os.path.join(out_dir, k), map_location=device))
        model_dict[k] = v
    return model_dict


class MinEdit:
    """
    A solver to determine the minimum number of indexes to convert an unsorted array into a sorted one
    """

    def __init__(self, num_items, range):
        self.n = num_items
        self.m = range

    def _validate(self, l):
        assert l.shape == (self.n,)
        assert (l < self.m).sum() == self.n

    def _rec_solve(self, l, i, upper_bound):
        if i < 0:
            return []
        x1 = self._rec_solve(l, i - 1, upper_bound)
        x1.append(i)
        if l[i] > upper_bound:
            return x1
        x2 = self._rec_solve(l, i - 1, l[i])
        return x1 if len(x1) <= len(x2) else x2

    def solve(self, l):
        input = np.array(l)
        self._validate(input)
        return self._rec_solve(input, self.n - 1, self.m)


def load_segmentation_model(model_dir,epoch_num=4):

    model = build_model(2)
    model_key = f"epoch_{epoch_num}/pytorch_model.bin"
    model_dict = load({model_key: model},model_dir)
    model = model_dict[model_key]
    model = model.to(device)
    model.eval()
    return model


def get_image_panels(model, img_path):

    assert os.path.exists(img_path)
    pil_img = P.Image.open(img_path, "r").convert("RGB")
    w, h = pil_img.size
    # print("H, W: ", h,w)
    # print("Here is the strip:")

    tensor_img = T.PILToTensor()(pil_img)
    preprocess = T.Lambda(lambda x: x / 256.0)

    batch = preprocess(tensor_img)[None]
    batch = batch.to(device)
    # batch = torch.Tensor(batch).to(device)
    outp = model(batch)
    prediction = outp[0]
    boxes = prediction["boxes"].detach().cpu().tolist()

    def get_alignment_score(box):
        bin_ht = h / 5
        bin_idx = int(box[1] / bin_ht)
        return box[0] + w * bin_idx

    boxes = sorted(boxes, key=get_alignment_score)
    # pprint(boxes)

    panels = [pil_img.crop(box) for box in boxes]

    return panels, prediction, tensor_img


def easyocr_bboxes(reader, img):
    transform = T.Lambda(lambda x: x) if img.shape[0] == 1 else T.Grayscale()
    ocr_img = transform(img).numpy().squeeze()
    bound = reader.readtext(ocr_img)
    bboxes = []
    panel_text = ""
    for pts, txt, conf in bound:
        bboxes.append([pts[0][0], pts[0][1], pts[2][0], pts[2][1]])
        panel_text += " " + txt.lower()
    bboxes = torch.Tensor(bboxes)
    # box_img = draw_bounding_boxes(T.PILToTensor()(img), boxes=bboxes, colors="red", width=1)
    # box_img = to_pil_image(box_img)
    return bboxes, panel_text


class Segmentation:
    def __init__(self,model_dir):
        self.model = load_segmentation_model(model_dir)
        self.reader = easyocr.Reader(["en"])
        self.ocr_metadata = []
        self.problems = []

    def process(
        self,
        strip,
        idx,
        dump_dir=None,
        metadata_dump_path=None,
    ):

        try:
            return self.map_dialogue_to_panels(strip, idx, dump_dir, metadata_dump_path)
        except Exception as e:
            print(e)
            # @TODO: Remove the raise
            raise e
            self.problems.append(
                f"comic_name : {strip['comic_name']}, strip_sequence : {strip['strip_sequence']}"
            )

            with open("./problems_panel_segmentation.txt", "w+") as f:
                for p in self.problems:
                    f.write(p)
                    f.write("\n")
            return {"panel_ids": [-2]}  # ,'ocr_bbox':[]}

    def map_dialogue_to_panels(
        self, strip, idx, dump_dir=None, metadata_dump_path=None
    ):
        if not os.path.exists(make_img_path(strip)):
            return {"panel_ids": [-1]}
        img_path = make_img_path(strip)
        panels, predictiodialon, tensor_img = get_image_panels(self.model, img_path)
        num_panels = len(panels)
        num_dialogues = len(strip["dialogues"])
        if num_panels == 0:

            panels = [to_pil_image(tensor_img)]
        if dump_dir is not None:
            data_path = os.path.join(
                dump_dir,
                strip["comic_name"].split("_")[0],
                strip["strip_sequence"].split(" ")[-1],
            )
            os.makedirs(data_path, exist_ok=True)
            [
                panel.convert("RGB").save(os.path.join(data_path, f"{i}.jpg"))
                for i, panel in enumerate(panels)
            ]

        strip_bboxes = []
        strip_text = []
        panel_scores = []
        for panel in panels:
            bboxes, panel_text = easyocr_bboxes(self.reader, T.PILToTensor()(panel))
            strip_bboxes.append(bboxes.tolist())
            strip_text.append(panel_text)
            dialogue_scores = [
                fuzz.partial_ratio(dialog.lower(), panel_text)
                for dialog in strip["dialogues"]
            ]

            # print(panel_text)
            # print(dialogue_scores)
            panel_scores.append(dialogue_scores)

        self.ocr_metadata.append(
            [strip["comic_name"], strip["strip_sequence"], strip_bboxes, strip_text]
        )

        panel_scores = np.array(panel_scores)
        if num_dialogues == 0:
            return {"panel_ids": [-1]}  # ,'ocr_bbox':[]}
        if num_panels == 0:
            return {"panel_ids": [-1]}  # ,'ocr_bbox':[]}

        if panel_scores.shape != (num_panels, num_dialogues):
            print(
                "***** PROBLEM:",
                strip["comic_name"],
                strip["strip_sequence"],
                panel_scores.shape,
                num_panels,
                num_dialogues,
            )
            return {"panel_ids": [-1]}  # ,'ocr_bbox':[]}
        panel_idxs = np.argmax(panel_scores, axis=0)

        # print("Panel idxs: ", panel_idxs)

        min_edit = MinEdit(num_dialogues, num_panels)
        edit_idxs = min_edit.solve(panel_idxs)
        # print("Editing idxs: ", edit_idxs)
        for _idx in edit_idxs:
            lb = 0 if _idx == 0 else panel_idxs[_idx - 1]
            ub = num_panels - 1 if _idx == num_dialogues - 1 else panel_idxs[_idx + 1]

            panel_idxs[_idx] = int((ub + lb) / 2)
        # print("Edited idxs: ", panel_idxs)
        if idx % 2000 == 0 and metadata_dump_path is not None:
            self.dump_metadata(metadata_dump_path)

        return {"panel_ids": panel_idxs}  # ,'ocr_bbox':strip_bboxes}

    def dump_metadata(self, path):
        df = pd.DataFrame(
            self.ocr_metadata,
            columns=["comic_name", "strip_sequence", "ocr_bboxes", "ocr_text"],
        )
        df.to_pickle(path)
        # print("Dumped!")
