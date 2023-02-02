import re 
import pandas as pd
import os 
import random
import jsonlines

def flatten_json(data: dict) -> dict:
    """ recursive flatten json elements from https://www.geeksforgeeks.org/flattening-json-objects-in-python/"""
    out = {}

    def flatten(x, name=""):
        # If the Nested key-value
        # pair is of dict type
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + "_")

        # If the Nested key-value
        # pair is of list type
        elif type(x) is list:
            i = 0

            for a in x:
                flatten(a, name + str(i) + "_")
                i += 1
        else:
            out[name[:-1]] = x

    flatten(data)
    return out


def construct_report(string: str) -> tuple:

    # normalize sections
    keywords = [x.replace(":","").lower() for x in re.findall("[A-Z0-9][A-Z0-9. ]*:",string)]

    # normalize sections
    paragraphs = re.findall("(\w+)*: *(.*?)(?=\s*(?:\w+:|$))", string.lower())
    sections = []
    for header, paragraph in paragraphs:
        if header in [x.replace(" ","_").replace("/","_") for x in keywords]:
            sections.append(":".join([header, ". ".join([x.strip() for x in paragraph.split(". ") if x])]))
        else:
            sections.append(" - ".join([header, ". ".join([x.strip() for x in paragraph.split(". ") if x])]))
    sections = list(map(lambda a: a + "." if a[-1] != "." else a, sections))
    paragraphs = re.findall("(\w+) *: *(.*?)(?=\s*(?:\w+:|$))", "  ".join(sections))

    report = {}
    for header, paragraph in paragraphs:
        sentence = paragraph.replace("  ", ".  ").replace("..", ".").replace(" - ."," - ")
        sentence = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", sentence)
        sentence = [x.strip() for x in sentence if len(x) > 2]
        report[header.replace("_", " ")] = [x.replace("_", " ") for x in sentence]
    report = flatten_json(report)
    topic = [x.split("_")[0] for x in report.keys()]
    body = [x for x in report.values()]
    report = pd.DataFrame(list(zip(topic, body)))
    try:
        report.columns = ["paragraph", "sentence"]
        report["ranking"] = report.index
        report["screen"] = report["sentence"].apply(lambda x: 1 if 'interval change' in x or 'compar' in x or 'prior' in x or 'improved from' in x else 0)
        reason = re.sub(" +", " ", " ".join([": ".join([key, value]) for (key,value) in collapse_report(report).items() if key in ['indication','history']]))
        text = re.sub(" +", " ", " ".join([": ".join([key, value]) for (key,value) in collapse_report(report[report.screen==0]).items() if key in ['findings','impression']]))
        if 'findings' in text and 'impression' in text:
            return reason, text 
        else:
            return None, None
    except ValueError:
        return None, None

# take a report dataframe and return a dictionary of the paragraphs
def collapse_report(report: pd.DataFrame) -> dict:
    """take raw text and return paragraphs in sections as key:value pairs"""
    out = pd.merge(
        report['paragraph'].drop_duplicates(),    
        report.groupby(['paragraph'])['sentence'].transform(lambda x: '  '.join(x)).drop_duplicates(),
        left_index=True,
        right_index=True
    )
    structure = dict()
    for index, row in out.iterrows():
        structure[row['paragraph']] = row['sentence']
    return structure


def extract_transform(row: dict) -> None:

    report_root = "./physionet.org/files/mimic-cxr/2.0.0/files"
    image_root = "./physionet.org/files/mimic-cxr-jpg/2.0.0/files"

    try:
        scans = os.listdir(os.path.join(image_root,row["part"],row["patient"]))
        scans = [x for x in scans if 'txt' not in x]
        for scan in scans:
            report = os.path.join(report_root,row["part"],row["patient"],scan+".txt")
            if os.path.exists(report):
                with open(report,"r") as f:
                    original = f.read()
                transformed = re.sub(" +"," ",original.replace("FINAL REPORT","").strip().replace("\n \n",".").replace("\n"," ")).replace(" . "," ").replace("..",".").replace("CHEST RADIOGRAPHS."," ").strip()
                if len(transformed) > 0:
                    reason, text = construct_report(transformed)
                    images = [os.path.join(image_root,row["part"],row["patient"],scan,x) for x in os.listdir(os.path.join(image_root,row["part"],row["patient"],scan))]
                    images = [x for x in images if os.path.exists(x)]
                    random.shuffle(images) # shuffle so we can reasonably sample 1 image per study
                    with jsonlines.open("dataset.jsonl","a") as writer:
                        for image in images:
                            writer.write({
                                "fold": row["patient"][0:3],
                                "image": image,
                                "study": image.split("/")[-2],
                                "original": transformed,
                                "report": report,
                                "patient": row["patient"],
                                "reason": reason, 
                                "text": " ".join([reason,text]) if reason is not None and text is not None else None
                            })
    except FileNotFoundError:
        pass