import streamlit as st
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

processor = BlipProcessor.from_pretrained("nathansutton/generate-cxr")
model = BlipForConditionalGeneration.from_pretrained("nathansutton/generate-cxr")

def humanize_report(report: str) -> str:
    report = report.replace("impression :","IMPRESSION:\n").replace("findings :","FINDINGS:\n").replace("indication :","INDICATION:\n")
    sentences = [x.split("\n") for x in report.split(".") if x]
    sentences = [item for sublist in sentences for item in sublist]
    sentences = [x.strip().capitalize() if ':' not in x else x for x in sentences]
    return ".  ".join(sentences).replace(":.",":").replace("IMPRESSION:","\n\nIMPRESSION:\n\n").replace("FINDINGS:","\n\nFINDINGS:\n\n").replace("INDICATION:","INDICATION:\n\n")

indication = st.text_input("What is the indication for this study")
img_file_buffer = st.file_uploader("Upload a single view from a Chest X-Ray (JPG preferred)")
if img_file_buffer is not None and indication is not None:

    image = Image.open(img_file_buffer)
    st.image(image, use_column_width=True)
    inputs = processor(
        images=Image.open(img_file_buffer), 
        text='indication:' + indication,
        return_tensors="pt"
    )
    output = model.generate(**inputs,max_length=512)
    report = processor.decode(output[0], skip_special_tokens=True)
    st.write(humanize_report(report))


