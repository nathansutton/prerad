import os 
import jsonlines
from concurrent.futures import ThreadPoolExecutor
from common import extract_transform

def run():

    # remove previous executions
    if os.path.exists("/opt/physionet/dataset.jsonl"):
        os.remove("/opt/physionet/dataset.jsonl")

    if os.path.exists("/opt/physionet/control.jsonl"):
        os.remove("/opt/physionet/control.jsonl")

    # create a control dictionary
    root = "/opt/physionet/physionet.org/files/mimic-cxr/2.0.0/files"
    with jsonlines.open("/opt/physionet/control.jsonl","w") as writer:
        parts = os.listdir(root)
        for part in parts:
            patients = os.listdir(os.path.join(root,part))
            for patient in patients:
                scan = [x for x in os.listdir(os.path.join(root,part,patient))  if x.endswith('.txt')]
                writer.write({"part": part, "patient": patient,"scan": scan})         


    # parse each record
    with ThreadPoolExecutor(max_workers=4) as executor:
        with jsonlines.open("/opt/physionet/control.jsonl","r") as reader:
            executor.map(extract_transform, reader)


# only run it if there are files downloaded
if __name__ == "__main__":
    try:
        if len(os.listdir('/opt/physionet/physionet.org/files/mimic-cxr/2.0.0/files')) > 0: 
            run()   
    except OSError:
        print("not downloaded yet")
