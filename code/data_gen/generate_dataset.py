import re
import requests
import shutil
import os
from bs4 import BeautifulSoup
import json
from segment import Segmentation
from datasets import Dataset
import datasets
import pandas as pd
from utils import replace_dataset, transcript_segment

pattern = r"^(.*?)\/([^\/]+)\/(\d\d\d\d\/\d\d\/\d\d)$"
database = {}


def extract(s):
    splits = re.split("(\S+):", s)[1:]
    speaker, dialog = [], []
    for i in range(1, len(splits), 2):
        speaker.append(splits[i - 1].strip().lower())
        dialog.append(splits[i].strip())
    return speaker, dialog


def get_data(url, visual_base_dir, idx, split="all",domain='seen'):
    try:
        match_ = re.match(pattern, url)
        comic_name, date1 = match_[2], match_[3]
        # if comic_name not in database:
        # database[comic_name] = {}
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        # meta_image = soup.find('meta', property = "og:image")
        meta_transcript = soup.find("meta", property="og:description")
        clean_transcript = meta_transcript["content"]
        date2 = date1.replace("/", "-")
        if not clean_transcript:
            print(f"{date1} {comic_name} Empty")
            database[idx] = {
                # database[comic_name][date2] = {
                "original_transcript": "",
                "comic_name": comic_name,
                "strip_sequence": f"ID: {date2}",
                "image_path": "",
                "speakers" : [''],
                "dialogues" : [''],
                "split" : split,
                'domain':domain
            }
            return False
        link = soup.select(".item-comic-image > img:nth-of-type(1)")[0]
        image_url = link.get("src")

        save_path = os.path.join(
            visual_base_dir,
            comic_name,
        )
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, date2 + ".jpg")

        r = requests.get(image_url, stream=True)
        if r.status_code == 200:
            r.raw.decode_content = True
            with open(save_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        else:
            print(f"{date1} {comic_name} Image Empty")
            return False

        # fix extraction

        # database[comic_name][date2] = {
        database[idx] = {
            "original": clean_transcript,
            "comic_name": comic_name,
            "strip_sequence": f"ID: {date2}",
            "image_path": save_path,
            "speakers" : [''],
                "dialogues" : [''],
                "split" : split,
                'domain':domain
        }
        return True
    except Exception as e:
        print(e)
        print(date1 + " error", comic_name)
        return False


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="ComSet")
    parser.add_argument("--urls", type=str, help="Path to ComSet Directory")
    parser.add_argument("--model_dir", type=str, help="Path to model directory")
    # if len(sys.argv) > 2:
    #     for file_name in sys.argv[2:]:
    #         temp_database = json.load(open(file_name, "r"))
    #         database = temp_database
    args = parser.parse_args()
    os.makedirs(os.path.join(args.base_dir), exist_ok=True)
    for domain in ["seen", "unseen"]:
        for split in ["train", "test", "validation", "all"]:
            database = {}
            if not os.path.exists(os.path.join(args.urls, domain, split + ".json")):
                continue
            visual_base_dir = os.path.join(args.base_dir, "visual", domain, split)
            url_list = json.load(
                open(os.path.join(args.urls, domain, split + ".json"), "r")
            )
            for idx, url in enumerate(url_list):
                get_data(url, visual_base_dir, idx, split, domain)
                if idx % 30 == 0:
                    print("Writing")
                    with open(
                        os.path.join(args.base_dir, "scraped_transcripts.json"), "w"
                    ) as openfile:
                        json.dump(database, openfile, indent=4, sort_keys=True)
            print("Writing")
            with open(
                os.path.join(args.base_dir, "scraped_transcripts.json"), "w"
            ) as openfile:
                json.dump(database, openfile, indent=4, sort_keys=True)

            comic_dataset = Dataset.from_pandas(pd.DataFrame(database).T)
            replace_dataset(comic_dataset, args.base_dir, split, domain)
            # Text Extraction from transcripts
    segmentor = Segmentation(args.model_dir)  
    transcript_segment(os.path.join(args.base_dir, "text"),args.base_dir,["train", "test", "validation", "all"],["seen", "unseen"])
    for domain in ["seen", "unseen"]:
        for split in ["train", "test", "validation", "all"]:        
            if not os.path.exists(os.path.join(args.base_dir, "text",domain,split)):
                continue   
            comic_dataset = datasets.load_from_disk(
                os.path.join(args.base_dir, "text",domain,split)
            )
            print(comic_dataset)
            # Extract BoundingBoxes
            comic_dataset = comic_dataset.map(
                segmentor.process,
                fn_kwargs={
                    "dump_dir": os.path.join(args.base_dir, "segmented",domain,split),
                },
                with_indices=True,
            )

            replace_dataset(comic_dataset, args.base_dir, split, domain)
