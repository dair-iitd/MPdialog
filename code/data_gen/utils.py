import re
import datasets
import os
import spacy
from copy import copy as cpy

nlp = spacy.load("en_core_web_sm")


def get_last_verb(doc):
    for t in reversed(list(doc)):
        if t.pos_ == "VERB":
            return t
    return None


def strip_spans(t: str, matches):
    s = ""
    prev = 0
    for m in matches:
        x, y = m.span()
        s += t[prev:x] + " "
        prev = y + 1
    s += t[prev:]
    return s


def get_proper_nouns(doc):
    return [t for t in doc if t.pos_ == "PROPN"]


def get_chain(doc, src):
    chain = []
    curr = src
    while True:
        chain.append(curr)
        if curr.dep_ == "ROOT":
            break
        curr = curr.head
    return chain


def get_nsubj(doc, verb_token):
    cands = []
    if verb_token is None:
        return cands
    for t in doc:
        if t.dep_ == "nsubj" and t.head in get_chain(doc, verb_token):
            cands.append(t)
    return cands


def match_speakers(doc, speaker_set):
    matches = []
    for t in doc:
        if t.text.lower() in speaker_set:
            matches.append(t)
    return matches


def extract_speaker(doc, speakers):
    # import pdb; pdb.set_trace()
    speaker = "other"
    pnouns = get_proper_nouns(doc)
    speaker_matches = match_speakers(doc, speakers)

    pnouns = list(set(pnouns).intersection(set(speaker_matches)))
    if len(pnouns) == 1:
        speaker = pnouns[0].text
        # if not speaker.isalpha():
        #   pprint((doc.text, speaker))
    else:
        end_verb = get_last_verb(doc)
        nsubjs = get_nsubj(doc, end_verb)
        if len(nsubjs) == 0:
            nsubjs = get_nsubj(doc, list(doc)[-1])

        nsubjs = list(set(nsubjs).intersection(set(speaker_matches)))
        if len(nsubjs) == 0:
            if len(speaker_matches) > 0:
                # pprint(("speaker matches: ", doc.text, speaker_matches))
                nsubjs = speaker_matches[:1]
        if len(nsubjs) > 0:
            speaker = nsubjs[0].text
        else:
            # speaker = 'other'
            pass

    return speaker.lower()


def fetch_speakers(ex, global_speakers, replacements=[]):

    s_list, d_list = strip_dialogues(ex["original"])
    if len(s_list) == 0:
        _reparsed = reparse(ex, replacements=replacements)
        s_list, d_list = _reparsed["speakers"], _reparsed["dialogues"]
    dialogues = d_list
    speakers = []
    for s in s_list:
        if len(speakers) == len(dialogues):
            break
        doc = nlp(s)
        speaker = extract_speaker(doc, global_speakers)
        speakers.append(speaker)

    return {"dialogues": dialogues, "speakers": speakers}


def merge_speakers(ex, merge_from, merge_to):
    speakers = ex["speakers"]
    for i in range(len(speakers)):
        for mf, mt in zip(merge_from, merge_to):
            if speakers[i] in set(mf):
                speakers[i] = mt
    return {"speakers": speakers}


def add_split(ex, split):
    ex["split"] = split
    return ex


def rm_split(ex):
    del ex["split"]
    return ex


def replace_dataset(ds, base_dir, splits=None, domain=None):
    if splits is None:
        splits = set(ds["split"])
        ds_dict = datasets.DatasetDict(
            {
                mode: ds.filter(lambda x: x["split"] == mode).map(rm_split)
                for mode in splits
            }
        )
        # ds_dict.save_to_disk(dataset_conf["dir"])
        ds_dict.save_to_disk(os.path.join(base_dir, "text"))
    else:

        ds.save_to_disk(os.path.join(base_dir, "text", domain, splits))


def replace_comic(original_ds, replaced_comic, comic_name):
    original_stripped = original_ds.filter(lambda x: comic_name not in x["comic_name"])
    final_ds = datasets.concatenate_datasets([original_stripped, replaced_comic])
    return final_ds


def ds_difference(ds1, ds2, key="strip_sequence"):
    keys = set(ds2[key]) if len(ds2) else set()
    return ds1.filter(lambda x: x[key] not in keys)


def get_speaker_stats(ds, sort=False):
    sp_set = {}
    for x in ds:
        for sp in x["speakers"]:
            sp_set[sp] = sp_set.get(sp, 0) + 1
    if sort:
        return sorted(sp_set.items(), key=lambda x: -x[1])
    return sp_set


def  strip_dialogues(text):
    import re

    t = cpy(text)
    s_temp = [x.strip() for x in t.split(":") if x.strip() != ""]
    s_list, d_list = [], []
    regex = r'"(.*?)"'
    for i, s in enumerate(s_temp):
        matches = re.findall(regex, s)
        if i == 0:
            s_list.append(s)
            continue
        if len(matches) == 0:
            continue

        d_list.append(". ".join(matches))
        s_new = strip_spans(s, re.finditer(regex, s)).strip()
        if s_new != "":
            s_list.append(s_new)

    min_l = min(len(s_list), len(d_list))
    return s_list[:min_l], d_list[:min_l]


def get_bad_sp(ds, thresh=6):
    sp_set = get_speaker_stats(ds)
    bad_sp = set()
    for k, v in sp_set.items():
        if v <= thresh:
            bad_sp.add(k)
    return bad_sp


def make_others(ex, others):
    speakers = ex["speakers"]
    for i in range(len(speakers)):
        if speakers[i] in others:
            speakers[i] = "others"
    ex["speakers"] = speakers
    return ex


def get_url(obj, comic_name=None):
    comic_name = (
        comic_name
        if comic_name is not None
        else obj["comic_name"].replace("_final", "")
    )
    y, m, d = obj["strip_sequence"].split(":")[-1].strip().split("-")
    return f"https://www.gocomics.com/{comic_name}/{y}/{m}/{d}"


def flatten(x):
    res = []
    for xs in x:
        res += xs
    return res


def extract(s):
    splits = re.split("([a-zA-Z_]+):", s)[1:]
    speaker, dialog = [], []
    for i in range(1, len(splits), 2):
        dialo = splits[i].strip()
        if dialo == "":
            continue
        speaker.append(splits[i - 1].strip().lower())
        dialog.append(splits[i].strip())
    return speaker, dialog


def reparse(ex, replacements):
    t = ex["original"]
    for orig, repl in replacements:
        t = re.sub(orig, repl, t, flags=re.IGNORECASE)
    speakers, dialogues = extract(t)
    data = {k: v for k, v in ex.items()}
    data["speakers"] = speakers
    data["dialogues"] = dialogues
    return data


def transcript_segment(comic_all_path,base_dir, splits, domains):
    comic_all = None
    for domain in domains:
        for split in splits:
            if not os.path.exists(os.path.join(base_dir, "text",domain,split)):
                continue 
            temp = datasets.load_from_disk(
                os.path.join(base_dir, "text",domain,split)
            )
            if comic_all is None:
                comic_all = temp
            else:
                comic_all = datasets.concatenate_datasets([comic_all, temp])

    # comic_all = datasets.load_from_disk(comic_all_path)
    comic_all = process_getfuzzy(comic_all, base_dir, split, domain)
    comic_all = process_garfield(comic_all, base_dir, split, domain)
    comic_all = process_familytree(comic_all, base_dir, split, domain)
    comic_all = process_cleats(comic_all, base_dir, split, domain)
    comic_all = process_inkpen(comic_all, base_dir, split, domain)
    comic_all = process_cathy(comic_all, base_dir, split, domain)
    comic_all = process_doonesbury(comic_all, base_dir, split, domain)
    comic_all = process_calvinhobbes(comic_all, base_dir, split, domain)
    comic_all = process_heartofthecity(comic_all, base_dir, split, domain)
    comic_all = process_peanuts(comic_all, base_dir, split, domain)
    comic_all = process_rip(comic_all, base_dir, split, domain)
    comic_all = process_bigtop(comic_all, base_dir, split, domain)
    comic_all = process_bignate(comic_all, base_dir, split, domain)
    
    splits = set(comic_all["split"])
    ds_dict = datasets.DatasetDict(
        {
            'seen': datasets.DatasetDict(
                            {
                                mode: comic_all.filter(lambda x: x["split"] == mode).map(rm_split)
                                for mode in splits if mode!='all'
                            }
            ),
            'unseen':datasets.DatasetDict(
                            {
                                mode: comic_all.filter(lambda x: x["split"] == mode).map(rm_split)
                                for mode in splits if mode=='all'
                            }
            )
        }
    )
    # ds_dict.save_to_disk(dataset_conf["dir"])
    ds_dict.save_to_disk(os.path.join(base_dir, "text"))
    # comic_all.map(lambda x: x).save_to_disk(comic_all_path)
    # return comic_all



###Inkpen


def process_inkpen(comic_all, base_dir, split, domain):
    inkpen_ds = comic_all.filter(lambda x: "inkpen" in x["comic_name"])
    repls = [
        (r"\bJenn Erica:", "Jenn_Errica:"),
        (r"\bMoxie Gumption:", "Moxie:"),
        (r"\bCaptain Victorious:", "Captain_Victorious:"),
        (r"\bCaptan Victorious:", "Captain_Victorious:"),
        (r"\bCaptain Victory:", "Captain_Victorious:"),
        (r"\bRick Wrigley:", "rick_wrigley:"),
        (r"\bscrappy lad:", "scrappy_lad:"),
        (r"\bRick:", "rick_wrigley:"),
        (r"\bcaptain:", "Captain_Victorious:"),
        (r"\bjenn:", "Jenn_Errica:"),
        (r"\bLad:", "scrappy_lad:"),
        (r"\bScrappy:", "scrappy_lad:"),
        (r"\bdynman:", "dynaman:"),
        (r"\bdyanman:", "dynaman:"),
        (r"\bvikign:", "viking:"),
        (r"\bralson:", "ralston:"),
        (r"\bhamhocl:", "hamhock:"),
        (r"\bhamhcok:", "hamhock:"),
        (r"\bhamhoc:", "hamhock:"),
        (r"\bfrtiz:", "fritz:"),
        (r"\bVictory:", "Captain_Victorious:"),
    ]
    inkpen_reparsed = inkpen_ds.map(reparse, fn_kwargs={"replacements": repls})
    inkpen_reparsed = inkpen_reparsed.map(
        make_others,
        fn_kwargs={
            "others": [
                "man",
                "woman",
                "cop",
                "elmo",
                "flower",
                "clong",
                "leprechaun",
                "burglar",
                "mailman",
                "witch",
                "lion",
                "rick_wrigley",
                "kid",
                "wife",
                "girl",
                "husband",
                "son",
                "lawyer",
                "owl",
                "dragon",
                "dino",
                "dinoman",
                "monster",
                "bucket",
                "snowman",
                "voices",
                "brain",
                "ted",
                "note",
                "soldier",
                "bear",
                "scientist",
                "children",
                "phone",
                "bartender",
                "turtle",
                "cat",
                "boy",
                "noise",
                "voice",
                "director",
                "pig",
                "wolf",
                "dog",
                "villain",
                "dentist",
                "brett",
                "doctor",
                "judge",
                "mom",
                "enemies",
                "robot",
                "shrimptom",
                "squirrel",
                "clerk",
                "lizard",
                "ox",
                "gravestone",
                "fox",
                "bird",
                "puma",
                "weasel",
                "rowdy",
                "clamwich",
                "jeremy",
            ]
        },
    )
    bad_sp = get_bad_sp(inkpen_reparsed, thresh=65)
    inkpen_cleaned = inkpen_reparsed.filter(
        lambda x: len(set(x["speakers"]).intersection(bad_sp)) == 0
    )
    return replace_comic(comic_all, inkpen_cleaned, "inkpen")
    # replace_dataset(inkpen_corrected, base_dir, split, domain)


def process_cathy(comic_all, base_dir, split, domain):
    train_ds = comic_all
    cathy_comics = train_ds.filter(lambda x: "cathy" in x["comic_name"])
    # pprint(cathy_comics)
    # pprint(get_speaker_stats(cathy_comics, sort=True))

    cathy_reparsed = cathy_comics.map(
        reparse,
        fn_kwargs={
            "replacements": [
                (r"Mrs. Hillman", "Mrs_Hillman"),
                (r"Mr. Hillman", "Mr_Hillman"),
                (r"(\(.*?\))", ""),
            ]
        },
    )
    bad_sp = get_bad_sp(cathy_reparsed, thresh=10)
    cathy_cleaned = cathy_reparsed.filter(
        lambda x: len(set(x["speakers"]).intersection(bad_sp)) == 0
    )
    cathy_all = cathy_cleaned.map(
        make_others,
        fn_kwargs={"others": ["man", "woman", "assistant", "caption", "phone", "then"]},
    )
    return replace_comic(comic_all, cathy_all, "cathy")
    # replace_dataset(cathy_corrected, base_dir, split, domain)

def merge(ex,speakers):
    keys = ex.keys()
    merge_from = [['buck', 'buckys'], ['satche', 'satchlel', 'sathel'], ['robs', 'robert', 'robs']]
    merge_to = ['bucky', 'satchel', 'rob']
    changed = False
    for mf, mt in zip(merge_from, merge_to):
        intersect = set(mf).intersection(ex['speakers'])
        if len(intersect) == 0:
            continue
        changed = True
        for sp in intersect:
            text = ex['original']
            s = ""
            prev = 0
            for it in re.finditer(fr'(\b{sp}\b)', text, re.IGNORECASE):
                x,y = it.span()
                s += text[prev:x] + mt
                prev = y
            s = text[prev:]
            ex['original'] = s  


    if changed:
        dial_sp = fetch_speakers(ex,speakers)
        
        dial_sp.update({'original': ex['original']})
        return dial_sp
    return {k: ex[k] for k in keys}

def process_getfuzzy(comic_all, base_dir, split, domain):
    fuzzy_ds = comic_all.filter(lambda x: x["comic_name"] == "getfuzzy")
    # repls = []
    # fuzzy_reparsed = fuzzy_ds.map(
    #     reparse, fn_kwargs={"replacements": repls}, load_from_cache_file=False
    # )
    
    '''illegals = set(['answers', 'asks', 'replies', 'says', ''])
    import pdb; pdb.set_trace()
    fuzzy_ill_ds = fuzzy_ds.filter(lambda x: len(set(y.lower() for y in x['speakers']).intersection(illegals)) != 0)
    if len(fuzzy_ill_ds):
        all_doc = []
        for text in fuzzy_ill_ds['original']:
            s_list, _ = strip_dialogues(text)
            for pretext in s_list:
                all_doc.append(nlp(pretext))
        speakers = set()
        for doc in all_doc:
            for t in doc:
                if t.dep_ == 'nsubj' and t.pos_ == 'PROPN' and t.text.isalpha() and len(t.text) > 2:
                    speakers.add(t.text.lower())
        speakers.add('caption')
        fuzzy_corrected = fuzzy_ill_ds.map(
            fetch_speakers, fn_kwargs={"global_speakers": speakers}
        )
        fuzzy_corrected = fuzzy_corrected.map(
            merge, fn_kwargs={"speakers": speakers}
        )
        fuzzy_corrected = fuzzy_corrected.filter(lambda x: len(x['dialogues']) >= 2)
        
        fuzzy_rem = ds_difference(fuzzy_ds, fuzzy_ill_ds)
        fuzzy_ds = datasets.concatenate_datasets([fuzzy_rem, fuzzy_corrected])
        import pdb; pdb.set_trace()'''
    # # #
    repls = [
    ('Bucky Katt:', 'Bucky:'),
    ('Satchel Pooch:', 'Satchel:'),
    ('Rob Wilco:', 'Rob:'),                                                  
    ]
    def f(ex):
        for x in repls:
            if x[0].lower() in ex['original'].lower():
                return True
        return False

    fuzzy_ill_ds = fuzzy_ds.filter(f)

    repls += [
    ('birds:', 'birds'),
    ('philosophers:', 'philosophers'),
    ('maxim:', 'maxim'),
    ('it:', 'it'),
    ('something:', 'something'),
    ('two:', 'two'),
    ('adventure:', 'adventure'),
    ('universe:', 'universe'),
    (' sez:', ':'),
    ('everything:', 'everything'),
    ('scaloppine:', 'scaloppine'),
    ('correction:', 'correction'),
    ('mediums:', 'mediums'),
    ('moccasin:', 'moccasin'),
    ('wall:', 'wall'),
    ('gold:', 'gold'),
    ('straight:', 'straight'),
    (' Kat:', ':'),
    ('democrat:', 'democrat'),
    ('world:', 'world'),
    ('here:', 'here'),
    ('go:', 'go'),
    ('again:', 'again'),
    ('this:', 'this'),
    ('that:', 'that'),
    ('example:', 'example'),
    ('way:', 'way'),
    ('satchell:', 'satchel:'),
    ('bugs:', 'bugs'),
    ('mothman:', 'mothman'),
    ('me:', 'me'),
    ('one:', 'one'),
    ('Mac Manc McManx:', 'Mac_Manc_McManx:'),
    ('Joe Doman:', 'joe:')
    ]
    fuzzy_reparsed = fuzzy_ill_ds.map(reparse, fn_kwargs={'replacements': repls})

    fuzzy_reparsed = fuzzy_reparsed.map(make_others, fn_kwargs={'others': [
    'radio', 'note', 'ferret', 'machine', 'poodle', 'narrator', 'cat', 'dog', 'woman', 'saleswoman', 'vet', 'alex', 'officer', 'berger', 'waitress', 'todd', 'man', 'salesman'
    ]})
    bad_sp = get_bad_sp(fuzzy_reparsed, thresh=5)
    fuzzy_corrected = fuzzy_reparsed.filter(lambda x: len(set(x['speakers']).intersection(bad_sp)) == 0)
    
    fuzzy_rem = ds_difference(fuzzy_ds, fuzzy_ill_ds)
    fuzzy_final = datasets.concatenate_datasets([fuzzy_rem, fuzzy_corrected])
    bad_sp = get_bad_sp(fuzzy_final, thresh=5)
    fuzzy_ds = fuzzy_final.map(make_others, fn_kwargs={'others':[
    'other', 'caption'
    ] + list(bad_sp)
    })
    # # #
    illegals = set(['answers', 'asks', 'replies', 'says', ''])
    # import pdb; pdb.set_trace()
    fuzzy_ill_ds = fuzzy_ds.filter(lambda x: len(set(y.lower() for y in x['speakers']).intersection(illegals)) != 0)
    if len(fuzzy_ill_ds):
        all_doc = []
        for text in fuzzy_ill_ds['original']:
            s_list, _ = strip_dialogues(text)
            for pretext in s_list:
                all_doc.append(nlp(pretext))
        speakers = set()
        for doc in all_doc:
            for t in doc:
                if t.dep_ == 'nsubj' and t.pos_ == 'PROPN' and t.text.isalpha() and len(t.text) > 2:
                    speakers.add(t.text.lower())
        speakers.add('caption')
        fuzzy_corrected = fuzzy_ill_ds.map(
            fetch_speakers, fn_kwargs={"global_speakers": speakers}
        )
        fuzzy_corrected = fuzzy_corrected.map(
            merge, fn_kwargs={"speakers": speakers}
        )
        fuzzy_corrected = fuzzy_corrected.filter(lambda x: len(x['dialogues']) >= 2)
        
        fuzzy_rem = ds_difference(fuzzy_ds, fuzzy_ill_ds)
        fuzzy_ds = datasets.concatenate_datasets([fuzzy_rem, fuzzy_corrected])
    return replace_comic(comic_all, fuzzy_ds, 'getfuzzy')


def process_doonesbury(comic_all, base_dir, split, domain):
    dons_all = comic_all.filter(
        lambda x: "doonesbury" in x["comic_name"], load_from_cache_file=False
    )
    dons_temp = dons_all.map(
        reparse,
        fn_kwargs={
            "replacements": [
                (r"\bb.d:", "b_d:"),
                (r"\bb.d.:", "b_d:"),
                (r"\bbd:", "b_d:"),
                (r"\bj.j:", "j_j:"),
                (r"\bjj:", "j_j:"),
                (r"\bsal:", "benjamin_doonesbury:"),
                (r"\bbenjy:", "benjamin_doonesbury:"),
                (r"\bbenjamin:", "benjamin_doonesbury:"),
                (r"\bwidow d:", "daisy_doonesbury:"),
                (r"\bwidow doonesbury:", "daisy_doonesbury:"),
                (r"\bmrs. doonesbury:", "daisy_doonesbury:"),
                (r"\bdaisy:", "daisy_doonesbury:"),
                (r"\bwoman \d+:", "woman"),
                (r"\bman \d+:", "man"),
                (r"\bjimmy carter:", "jimmy_carter:"),
                (r"\bcarter:", "jimmy_carter:"),
                (r"\bjimmy:", "jimmy_thudpucker:"),
                (r"\bthudpucker:", "jimmy_thudpucker:"),
                (r"\bmarilou:", "marilou_slackmeyer:"),
                (r"\bmrs. slackmeyer:", "marilou_slackmeyer:"),
                (r"\bmr. slackmeyer:", "phil_slackmeyer:"),
                (r"\bslackmeyer:", "phil_slackmeyer:"),
                (r"\bphil:", "phil_slackmeyer:"),
                (r"\bmr. harris:", "nate_harris:"),
                (r"\bnate harris:", "nate_harris:"),
                (r"\bnate:", "nate_harris:"),
                (r"\bmrs. harris:", "amy_harris:"),
                (r"\bamy harris:", "amy_harris:"),
                (r"\bamy:", "amy_harris:"),
                (r"\bzeke brenner:", "zeke_brenner:"),
                (r"\bzeke:", "zeke_brenner:"),
                (r"\balix:", "alex:"),
                (r"\bgeorge bush:", "george_bush:"),
                (r"\bgeorge bush:", "george_bush:"),
                (r"\bbarbara bush:", "boopsie:"),
                (r"\bbush:", "george_bush:"),
                (r"\bgeorge:", "george_bush:"),
                (r"\bdonald trump:", "donald_trump:"),
                (r"\btrump:", "donald_trump:"),
                (r"\bwarren buffett:", "warren_buffett:"),
                (r"\bbuffett:", "warren_buffett:"),
                (r"\bwarren:", "warren_buffett:"),
                (r"\bDan Quayle:", "dan_quayle:"),
                (r"\bBen Quayle:", "ben_quayle:"),
                (r"\bQuayle:", "dan_quayle:"),
                (r"\bRonald Reagan:", "ronald_reagan:"),
                (r"\bReagan:", "ronald_reagan:"),
                (r"\bronald:", "ronald_reagan:"),
                (r"\bBill Clinton:", "bill_clinton:"),
                (r"\bPresident Clinton:", "bill_clinton:"),
                (r"\bbarack obama:", "barack_obama:"),
                (r"\bobama:", "barack_obama:"),
                (r"\bNewt Gingrich:", "newt_gingrich:"),
                (r"\bGingrich:", "newt_gingrich:"),
                (r"\bnewt:", "newt_gingrich:"),
                (r"\bDavid Duke:", "david_duke:"),
                (r"\bGropenfuhrer/Arnold Schwarzenegger:", "arnold_schwarzenegger:"),
                (r"\bGropenfuhrer/Schwarzenegger:", "arnold_schwarzenegger:"),
                (r"\bArnold Schwarzenegger:", "arnold_schwarzenegger:"),
                (r"\bArnold:", "arnold_schwarzenegger:"),
                (r"\bSchwarzenegger:", "arnold_schwarzenegger:"),
                (r"\bdick cheney:", "dick_cheney:"),
                (r"\bcheney:", "dick_cheney:"),
                (r"\bbarbara:", "boopsie:"),
                (r"\bnguyen v. pham:", "phred:"),
                (r"\bnguyen v. phred:", "phred:"),
                (r"\bpham:", "phred:"),
                (r"\bmr. butts:", "mr_butts:"),
                (r"\bmr. buttsy:", "mr_butts:"),
                (r"\bmr. buttsw:", "mr_butts:"),
                (r"\bbutts:", "mr_butts:"),
                (r"\brev. sloan:", "sloan:"),
                (r"\bscot:", "sloan:"),
                (r"\bscott:", "sloan:"),
                (r"\bmr. kibbitz:", "sid_kibbitz:"),
                (r"\bkibbitz:", "sid_kibbitz:"),
                (r"\bsid:", "sid_kibbitz:"),
                (r"\bpresident king:", "president_king:"),
                (r"\buniversity president:", "president_king:"),
                (r"\bking:", "president_king:"),
                (r"\bpresident:", "president_king:"),
                (r"\bjim andrews:", "jim_andrews:"),
                (r"\bTiffany Andrews:", "tiffany_andrews:"),
                (r"\bjim:", "jim_andrews:"),
                (r"\bandrews:", "jim_andrews:"),
                (r"\bLacey Davenport:", "lacey:"),
                (r"\bdick Davenport:", "dick:"),
                (r"\bsam:", "samantha:"),
                (r"\bmr. jay:", "mr_jay:"),
                (r"\bjay:", "mr_jay:"),
                (r"\bray hightower:", "ray_hightower:"),
                (r"\bray:", "ray_hightower:"),
                (r"\bmel:", "melissa:"),
                (r"\bjeremy cavendish:", "jeremy_cavendish:"),
                (r"\bmr. cavendish:", "jeremy_cavendish:"),
                (r"\bjeremy:", "jeremy_cavendish:"),
                (r"\bmr. delacourt:", "duane_delacourte:"),
                (r"\bduane:", "duane_delacourte:"),
                (r"\bron headrest:", "ron_headrest:"),
                (r"\bheadrest:", "ron_headrest:"),
                (r"\bhenry kissinger:", "henry_kissinger:"),
                (r"\bkissinger:", "henry_kissinger:"),
                (r"\bhenry:", "henry_kissinger:"),
                (r"\boliver north:", "oliver_north:"),
                (r"\bnorth:", "oliver_north:"),
                (r"\bjerry brown:", "jerry_brown:"),
                (r"\bbrown:", "jerry_brown:"),
                (r"\bjerry:", "jerry_brown:"),
                (r"\bdan asher:", "dan_asher:"),
                (r"\basher:", "dan_asher:"),
                (r"\bdan doheny:", "dan_doheny:"),
                (r"\bdoheny:", "dan_doheny:"),
            ]
        },
    )
    others = [
        "aide",
        "tv",
        "voices",
        "television",
        "phone",
        "noise",
        "caller",
        "people",
        "voice",
        "person",
    ]
    bad_sp = get_bad_sp(dons_temp, thresh=72)
    bad_sp = bad_sp.union(set(others))

    dons_final = dons_temp.map(
        make_others, fn_kwargs={"others": list(bad_sp)}, load_from_cache_file=False
    )

    # d = get_speaker_stats(dons_final, sort=True)
    return replace_comic(comic_all, dons_final, "doonesbury")
    # replace_dataset(dons_corrected, base_dir, split, domain)


def process_familytree(comic_all, base_dir, split, domain):
    family_ds = comic_all.filter(lambda x: "familytree" in x["comic_name"])
    repls = {}
    family_reparsed = family_ds.map(reparse, fn_kwargs={"replacements": repls})
    bad_sp = get_bad_sp(family_reparsed, thresh=65)
    family_reparsed = family_reparsed.map(make_others, fn_kwargs={"others": bad_sp})

    def rm_quotes(ex):

        dialogues = ex["dialogues"]
        for i, s in enumerate(dialogues):
            res = re.findall(r',\s*[\'"](.*?)[\'"\.]?$', s)
            if len(res) == 0:
                ex["original"] = ""
                break
            else:
                dialogues[i] = res[0]
        ex["dialogues"] = dialogues
        return ex

    family_cleaned = family_reparsed.map(rm_quotes, load_from_cache_file=False)
    family_cleaned = family_cleaned.filter(lambda x: x["original"] != "")
    return replace_comic(comic_all, family_cleaned, "familytree")
    # replace_dataset(family_corrected, base_dir, split, domain)


# Calvin Hobbes

char_map_calvin = {
    "calvin": "calvin",
    "hobbes": "hobbes",
    "susie": "susie",
    "wormwood": "miss_wormwood",
    "spittle": "principal",
    "i can her mom now": "tbd",
    "mom": "mom",
    "dad": "dad",
    "calvn": "calvin",
    "cavlin": "calvin",
    "claivn": "calvin",
    "clavin": "calvin",
    "clvin": "calvin",
    "calin": "calvin",
    "calivn": "calvin",
    "voice": "voice",
    "hobes": "hobbes",
    "hotes": "hobbes",
    "roslyn": "rosalyn",
    "susis": "susie",
    "suzie": "susie",
}


def fix_ch(strip):
    m = strip["original"]
    speaker, dialogue = extract_2_ch(m) if f"\r\n" in m else extract_ch(m)
    return {
        "original": strip["original"],
        "speakers": speaker,
        "dialogues": dialogue,
        "comic_name": "calvinandhobbes",
        "split": strip["split"],
    }


def extract_ch(m):
    s = re.sub("(\(.*?\))", "", m)
    splits = re.split(r"(^|[!\.\?]|\r\n)\s*([a-zA-Z_\s/\-]+):", s)
    narr = "".join(splits[:2])
    speaker, dialogue = [], []
    if narr:
        speaker.append("narrator"), dialogue.append(narr)
    for i in range(2, len(splits), 3):
        if i + 2 >= len(splits):
            dialo = splits[i + 1].strip()
        else:
            dialo = (splits[i + 1] + splits[i + 2]).strip()
        if dialo == "":
            continue
        sp = splits[i].strip().lower()
        for i, f in char_map_calvin.items():
            if i in sp:
                sp = f
        speaker.append(sp)
        dialogue.append(dialo.strip('" '))
    return speaker, dialogue


def extract_2_ch(m):
    # global counter
    m_ = m.split("\r\n")
    # if len(m_)==1 and len(m.split(':'))>2: counter+=1
    speaker, dialogue = [], []
    for x in m_:
        if x.strip() == "":
            continue
        mm = re.split("(.*?):[^0-9]", x)
        if len(mm) == 1:
            sound = re.match("\*(.+)[\*!]", x.strip())
            if sound:
                sp, dia = "sound", sound[0].strip()
            else:
                sp, dia = "narrator", x.strip()
        elif len(mm) == 3:

            _, sp, dia = mm
            sp = sp.strip().lower()
            dia = dia.strip()
        elif x.find(":") > 0:
            sp = x[: x.find(":")].strip().lower()
            dia = x[x.find(":") + 1 :].strip()
        else:
            sp = "error_1"
            dia = x.strip()
        sp = re.sub(r"\(.*?\)", "", sp).strip()
        sp = re.sub(r"#?[0-9]+$", "", sp).strip()
        sp = re.sub(r"[0-9]\.", "", sp).strip()
        for i, f in char_map_calvin.items():
            if i in sp:
                sp = f
        speaker.append(sp), dialogue.append(dia.strip('" '))
    return speaker, dialogue


def process_calvinhobbes(comic_all, base_dir, split, domain):
    ch_all = comic_all.filter(
        lambda x: "calvinandhobbes" in x["comic_name"], load_from_cache_file=False
    )
    ch_all = ch_all.filter(lambda x: x["original"] != "")
    ch_all = ch_all.map(fix_ch, load_from_cache_file=False)
    bad_sp = get_bad_sp(ch_all, thresh=20)

    ch_final = ch_all.map(
        make_others, fn_kwargs={"others": list(bad_sp)}, load_from_cache_file=False
    )
    return replace_comic(comic_all, ch_final, "calvinandhobbes")
    # replace_dataset(ch_corrected, base_dir, split, domain)


##Cleats
def process_cleats(comic_all, base_dir, split, domain):
    cleats_ds = comic_all.filter(
        lambda x: "cleats" in x["comic_name"], load_from_cache_file=False
    )

    repls = [
        (r"(?<=[a-zA-Z])\s+#?[0-9]", ""),
        (r"Jack Dooley:", "jack_dooley:"),
        (r"gary Dooley:", "gary_dooley:"),
        (r"gary:", "gary_dooley:"),
        (r"deanna Dooley:", "deanna_dooley:"),
        (r"deanna:", "deanna_dooley:"),
        (r"deanne:", "deanna_dooley:"),
        (r"katie Dooley:", "katie_dooley:"),
        (r"katie:", "katie_dooley:"),
        (r"mr. Dooley:", "gary_dooley:"),
        (r"Jack:", "jack_dooley:"),
        (r"Mondo Ruiz:", "mondo_ruiz:"),
        (r"Mondo:", "mondo_ruiz:"),
        (r"Armando:", "mondo_ruiz:"),
        (r'Armando "Mondo" Ruiz:', "mondo_ruiz:"),
        (r"Abby Harper:", "abby_harper:"),
        (r"Abby:", "abby_harper:"),
        (r"Dr. James:", "dr_james:"),
        (r"Edith tippit:", "edith:"),
        (r"Edith:", "edith:"),
        (r"Coach Tippit:", "tippit_coach:"),
        (r"Tippit:", "tippit_coach:"),
        (r"Coach Nordling:", "nordling_coach:"),
        (r"mrs. Nordling:", "nordling_ki_mrs:"),
        (r"Nordling:", "nordling_coach:"),
        (r"Nordling\'s son:", "nordlings_son:"),
        (r"george oliver:", "george_oliver:"),
        (r"george:", "george_oliver:"),
        (r"jerome oliver:", "jerome_oliver:"),
        (r"jerome:", "jerome_oliver:"),
        (r"Georgie Wells:", "georgie_wells:"),
        (r"Georgie:", "georgie_wells:"),
        (r"Coach Georgie:", "georgie_wells:"),
        (r"Big Kat:", "katisha:"),
        (r"Kat:", "katisha:"),
        (r"Jack\'s Grandma:", "bertha:"),
        (r"Jack\'s gramma:", "bertha:"),
        (r"gramma:", "bertha:"),
        (r"referee:", "ref:"),
        (r"Mr. Fountoulakis:", "ref:"),
        (r"Josh:", "ref:"),
        (r"voice:", "noise:"),
        (r"grandfather:", "grandpa:"),
        (r"soccer ball:", "ball:"),
        (r"Peri\'s brother:", "michael:"),
        (r"[\( ]?out of frame[\)]?", ""),
        (r"raymond neely:", "neely:"),
        (r"mr. raymond neely:", "neely:"),
        (r"mr. neely:", "neely:"),
        (r"coach neely:", "neely:"),
        (r"raymond:", "neely:"),
    ]

    cleats_reparsed = cleats_ds.map(
        reparse, fn_kwargs={"replacements": repls}, load_from_cache_file=False
    )
    bad_sp = get_bad_sp(cleats_reparsed, thresh=100)
    bad_sp = bad_sp.union(set(["ball"]))
    cleats_cleaned = cleats_reparsed.map(
        make_others, fn_kwargs={"others": bad_sp}, load_from_cache_file=False
    )
    cleats_cleaned = cleats_cleaned.filter(lambda x: len(x["dialogues"]) > 0)
    return replace_comic(comic_all, cleats_cleaned, "cleats")
    # replace_dataset(cleats_corrected, base_dir, split, domain)


##Peanuts
def process_peanuts(comic_all, base_dir, split, domain):
    peanuts_ds = comic_all.filter(lambda x: x["comic_name"].startswith("peanuts"))
    repls = [
        (r"<BR>", ""),
        (r"\(BR\)", ""),
        (r"\bBrown:", "charlie_brown:"),
        (r"\bCharlie:", "charlie_brown:"),
        (r"\bPeppermint Patty:", "peppermint_patty:"),
        (r"\bPatty:", "peppermint_patty:"),
        (r"\bshroeder:", "schroeder:"),
    ]

    peanuts_reparsed = peanuts_ds.map(
        reparse, fn_kwargs={"replacements": repls}, load_from_cache_file=False
    )
    bad_sp = get_bad_sp(peanuts_reparsed, thresh=80)
    peanuts_cleaned = peanuts_reparsed.map(
        make_others, fn_kwargs={"others": bad_sp}, load_from_cache_file=False
    )
    peanuts_good = peanuts_reparsed.filter(
        lambda x: len(set(x["speakers"]).intersection(bad_sp)) == 0
        and len(x["speakers"]) > 0
    )
    peanuts_ill_ds = peanuts_reparsed.filter(
        lambda x: len(set(x["speakers"]).intersection(bad_sp)) != 0
        or len(x["speakers"]) == 0
    )
    all_doc = []
    if len(peanuts_ill_ds):
        for text in peanuts_ill_ds["original"]:
            s_list, _ = strip_dialogues(text)
            for pretext in s_list:
                all_doc.append(nlp(pretext))

    speakers = set()
    for doc in all_doc:
        for t in doc:
            if (
                t.dep_ == "nsubj"
                and t.pos_ == "PROPN"
                and t.text.isalpha()
                and len(t.text) > 2
            ):
                speakers.add(t.text.lower())
    # speakers.add('caption')
    speakers = speakers.union(set(list(get_speaker_stats(peanuts_cleaned).keys())))
    if "other" in speakers:
        speakers.remove("other")
    peanuts_corrected = peanuts_ill_ds.map(
        fetch_speakers, fn_kwargs={"global_speakers": speakers, "replacements": repls}
    )
    peanuts_corrected = peanuts_corrected.filter(lambda x: len(x["speakers"]) > 0)
    peanuts_corrected = peanuts_corrected.map(
        merge_speakers,
        fn_kwargs={
            "merge_from": [
                ["brown", "charlie", "charliee"],
                ["shroeder"],
                ["ace"],
                ["lucys", "lucyt"],
                ["patty"],
            ],
            "merge_to": [
                "charlie_brown",
                "schroeder",
                "snoopy",
                "lucy",
                "peppermint_patty",
            ],
        },
        load_from_cache_file=False,
    )
    peanuts_corrected_1 = datasets.concatenate_datasets(
        [peanuts_good, peanuts_corrected]
    )
    peanuts_cleaned = peanuts_corrected_1.map(
        make_others,
        fn_kwargs={"others": get_bad_sp(peanuts_corrected_1, thresh=9)},
        load_from_cache_file=False,
    )
    return replace_comic(comic_all, peanuts_cleaned, "peanuts")
    # replace_dataset(repl_ds, base_dir, split, domain)


# heartofthecity
def process_heartofthecity(comic_all, base_dir, split, domain):
    hoc_ds = comic_all.filter(lambda x: x["comic_name"].startswith("heartofthecity"))
    repls = [
        (r"(?<=[a-zA-Z])\s+#?[0-9]", ""),
        ("Mrs. Angelini:", "angelini:"),
        ("Miss Angelini:", "angelini:"),
        ("Mrs Angelini:", "angelini:"),
        ("Mrs. Aangelini:", "angelini:"),
        ("mrs. a:", "angelini:"),
        ("Male Character:", "man:"),
        ("Character with locket:", "character_:"),
        ("Male Personnel Character:", "man:"),
        ("Female Personnel Character:", "woman:"),
        ("Boy Character:", "boy:"),
        ("Delivery Character:", "delivery:"),
        ("Howard Zitman:", "zitman:"),
        ("Lady Charactaer:", "woman:"),
        ("Female Charactaer:", "woman:"),
        ("Morrie Character:", "other:"),
        ("Teacher Character:", "teacher:"),
        ("Gir Character:", "girl:"),
        ("Girl Character:", "girl:"),
        ("herat:", "heart:"),
        ("mother:", "mom:"),
        ("is:", "is"),
        ("self:", "self"),
        ("Clerk Character:", "clerk:"),
        ("Cableman Character:", "cableman:"),
        ("Dad Character:", "dad:"),
        ("/Characters:", ":"),
        ("Tall Character:", "character_:"),
        ("Masked Character:", "character_:"),
        ("Hugh Character:", "hugh:"),
        ("Swordsman Character:", "swordsman:"),
        ("Skull Character:", "character_:"),
        ("Registration Character:", "character_:"),
        ("Courier Character:", "character_:"),
        ("Space Alien Character:", "alien:"),
        ("Blackfaced Character:", "character_:"),
        ("Other Character:", "character_:"),
        ("character_:", "character:"),
        ("Heart;", "Heart:"),
        ("Dean;", "dean:"),
        ("mom;", "mom:"),
        ("(adult)", ""),
        ("Mrs. Teekle:", "teekle"),
    ]
    hoc_reparsed = hoc_ds.map(
        reparse, fn_kwargs={"replacements": repls}, load_from_cache_file=False
    )
    bad_sp = get_bad_sp(hoc_reparsed, thresh=50)
    bad_sp = bad_sp.union(
        set(
            [
                #   'ball'
            ]
        )
    )
    hoc_cleaned = hoc_reparsed.map(
        make_others, fn_kwargs={"others": bad_sp}, load_from_cache_file=False
    )
    return replace_comic(comic_all, hoc_cleaned, "heartofthecity")
    # replace_dataset(repl_ds, base_dir, split, domain)


##RipHaywire

charmap_rip = {
    "haywire": "rip haywire",
    "rip": "rip haywire",
    "rip haywire": "rip haywire",
    "cobra": "cobra carson",
    "rj": "rodeo jones",
    "r.j": "rodeo jones",
    "r.j.": "rodeo jones",
    "dutch": "dutch haywire",
    "lady": "woman",
    "cobra caron": "cobra carson",
    "rip carson": "rip haywire",
    "li'l rip haywire": "rip haywire",
    "tahini jones": "tahiti jones",
    "spur": "uncle spur",
    "cora carson": "cobra carson",
    "cobra jones": "cobra carson",
    "feng shui kelly": "fengshui kelly",
    "tnt's owner": "rip haywire",
    "jones": "rodeo jones",
    "rip haywire's mom": "dr. pain",
    "rip's mom": "dr. pain",
    "mom": "dr. pain",
    "cobra carlson": "cobra carson",
    "r,j,": "rodeo jones",
    "fred finkle": "frank finkle",
    "doctor": "doc",
}


def extract_rip(m):
    # global counter

    m_ = m.split("\r\n")

    # if len(m_)==1 and len(m.split(':'))>2: counter+=1
    speaker, dialogue = [], []
    for x in m_:
        if x.strip() == "":
            continue
        mm = re.split("(.*?):[^0-9]", x)
        if len(mm) == 1:
            sound = re.match("\*(.+)[\*!]", x.strip())
            if sound:
                sp, dia = "sound", sound[0].strip()
            else:
                sp, dia = "narrator", x.strip()
        elif len(mm) == 3:

            _, sp, dia = mm
            sp = sp.strip().lower()
            dia = dia.strip()
        elif x.find(":") > 0:
            sp = x[: x.find(":")].strip().lower()
            dia = x[x.find(":") + 1 :].strip()
        else:
            sp = "error_1"
            dia = x.strip()
        sp = re.sub(r"\(.*?\)", "", sp).strip()
        sp = re.sub(r"#?[0-9]+", "", sp).strip()
        if sp in charmap_rip:
            sp = charmap_rip[sp]
        speaker.append(sp), dialogue.append(dia)
    return speaker, dialogue




def fix_rip(strip):
    m = strip["original"]
    speaker, dialogue = extract_rip(m)
    return {
        "original": strip["original"],
        "speakers": speaker,
        "dialogues": dialogue,
        "comic_name": "riphaywire",
        "split": strip["split"],
    }


def process_rip(comic_all, base_dir, split, domain):
    rip_all = comic_all.filter(
        lambda x: "riphaywire" in x["comic_name"], load_from_cache_file=False
    )
    rip_all = rip_all.filter(lambda x: x["original"] != "")
    rip_all = rip_all.map(fix_rip, load_from_cache_file=False)
    bad_sp = get_bad_sp(rip_all, thresh=20)
    rip_final = rip_all.map(
        make_others, fn_kwargs={"others": list(bad_sp)}, load_from_cache_file=False
    )
    return replace_comic(comic_all, rip_final, "riphaywire")
    # replace_dataset(rip_corrected, base_dir, split, domain)


# Garfield

charmap_garfield = {
    "friend": "lyman",
    "john": "jon",
    "mouse": "squeak",
    "mailman": "herman",
    "doc boy": "docboy",
    "flea": "insect",
    "fly": "insect",
    "voice": "other",
}


def fix_garfield(strip):
    m = strip["original"]
    speaker, dialogue = extract_garfield(m)
    return {
        "original": strip["original"],
        "speakers": speaker,
        "dialogues": dialogue,
        "comic_name": "garfield",   
        "split": strip["split"],
    }


def extract_garfield(m):
    m_ = m.split("\r\n")
    speaker, dialogue = [], []
    for x in m_:
        if x.strip() == "":
            continue
        mm = re.split("(.*?):[^0-9]", x)
        if len(mm) == 1:
            sound = re.match("\*(.+)[\*!]", x.strip())
            if sound:
                sp, dia = "sound", sound[0].strip()
            else:
                sp, dia = "narrator", x.strip()
        elif len(mm) == 3:

            _, sp, dia = mm
            sp = sp.strip().lower()
            dia = dia.strip()
        elif x.find(":") > 0:
            sp = x[: x.find(":")].strip().lower()
            dia = x[x.find(":") + 1 :].strip()
        else:
            sp = "error_1"
            dia = x.strip()
        sp = re.sub(r"\(.*?\)", "", sp).strip()
        sp = re.sub(r"#?[0-9]+", "", sp).strip()
        if sp in charmap_garfield:
            sp = charmap_garfield[sp]
        speaker.append(sp), dialogue.append(dia)
    return speaker, dialogue


def process_garfield(comic_all, base_dir, split, domain):
    garfield_all = comic_all.filter(
        lambda x: "garfield" in x["comic_name"], load_from_cache_file=False
    )
    garfield_all = garfield_all.filter(lambda x: x["original"] != "")
    garfield_all = garfield_all.map(fix_garfield, load_from_cache_file=False)
    bad_sp = get_bad_sp(garfield_all, thresh=50)
    garfield_final = garfield_all.map(
        make_others, fn_kwargs={"others": list(bad_sp)}, load_from_cache_file=False
    )
    return replace_comic(comic_all, garfield_final, "garfield")
    # replace_dataset(garfield_corrected, base_dir, split, domain)


#Bigtop
def process_bigtop(comic_all, base_dir, split, domain):
    bigtop_ds = comic_all.filter(lambda x: "bigtop" in x["comic_name"])
    repls = []
    bigtop_reparsed = bigtop_ds.map(reparse, fn_kwargs={"replacements": repls})
    bad_sp = get_bad_sp(bigtop_reparsed, thresh=65)
    bigtop_reparsed = bigtop_reparsed.map(
        make_others, fn_kwargs={"others": list(bad_sp)},
    )
    bigtop_cleaned = bigtop_reparsed.filter(
        lambda x: len(set(x["speakers"]).intersection(bad_sp)) == 0
    )
    return replace_comic(comic_all, bigtop_cleaned, "bigtop")
    # replace_dataset(bigtop_corrected, base_dir, split, domain)
#bignate
def process_bignate(comic_all, base_dir, split, domain):
    bignate_ds = comic_all.filter(lambda x: "bignate" in x["comic_name"])
    repls = []
    bignate_reparsed = bignate_ds.map(reparse, fn_kwargs={"replacements": repls})
    bad_sp = get_bad_sp(bignate_reparsed, thresh=65)
    bignate_reparsed = bignate_reparsed.map(
        make_others, fn_kwargs={"others": list(bad_sp)},
    )
    bignate_cleaned = bignate_reparsed.filter(
        lambda x: len(set(x["speakers"]).intersection(bad_sp)) == 0
    )
    # replace_dataset(bignate_corrected, base_dir, split, domain)
    return replace_comic(comic_all, bignate_cleaned, "bignate")