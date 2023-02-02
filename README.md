# A new generative model for radiology

tl;dr you should include text inputs along with images

Machine learning in radiology has come a long way. For a long time the goal was simply to make a probability estimates of different conditions available to the radiologist at the time of interpretation. As evidence, see any of the hundreds of AI vendors that have commercialized computer vision algorithms. On the academic frontier, recent advances have made it possible to generate realistic sounding radiology reports directly from an image. The first paper I found describing such a model was from 2017, but there have been many more recently with the onset of transformers. However, every example I have found suffers from the same structural problem. 

__They aren't answering a clinical question.__

When a provider orders an imaging study the ordering provider is asking the radiologist to combine their education and experience to answer a clinical question. Unlike most specialist consultations, the format is asynchronous. Occasional telephone or face-to-face consults do occur for time-sensitive findings, but this is not the norm. Instead, the radiology report is the primary product of this provider-radiologist specialist consultation. This document's purpose to answer the clinical question, or indication. It usually has a succinct 'impression' section at the end. For example…

```
INDICATION: 36yo M with hypoxia // ?pna, aspiration.  
FINDINGS: PA and lateral views of the chest provided. The lungs are adequately aerated. There is a focal consolidation at the left lung base adjacent to the lateral hemidiaphragm. There is mild vascular engorgement. There is bilateral apical pleural thickening. The cardiomediastinal silhouette is remarkable for aortic arch calcifications. The heart is top normal in size.  
IMPRESSION: Focal consolidation at the left lung base, possibly representing aspiration or pneumonia. Central vascular engorgement.  
```

Medical images alone are not sufficient to answer why an imaging study was ordered by a provider. For example, a provider might order a chest x-ray because their patient is presenting with shortness of breath and they suspect pneumonia. In another case, they might suspect a fracture after a motor vehicle collision and order a chest x-ray to rule out a broken rib. The same image could answer either of these clinical questions.

In my literature search I found seven studies describing model architectures that could conditionally generate entire radiology reports from an image, but none had text inputs (Jing et al. 2017, Yuan et al. 2019, Miura et al. 2020, Fenglin et al. 2021, Sirshar et al. 2022, Chen et al. 2022, Yang et al. 2022).  This is a problem because all the variation in the generated reports will come from the images (thus, not answering a clinical question in posed in text by the ordering provider).

To realistically describe what a radiologist is doing when they write a report, we need to utilize the same inputs. This means the conditional generation of radiology reports should include both image and text inputs and have a text output. This year a new transformer architecture particularly suited for this type of multi-modal problem was just released by SalesForce (Li et al. 2022). BLIP has a dual text and vision encoder paired with a text decoder. This allows it to continue generating new text for a radiology report from a given prompt's starting point. Lucky for us, the first paragraph of most radiology reports is the clinical question!

Starting from the base BLIP image captioning model, I fine-tuned a causal language model to generate radiology reports from a chest x-ray and a small prompt. 

Does it work? Let's go back to our original radiology report and perturb it with two different clinical indications. On the left we show the original question for this image ('question pneumonia') and on the right a fictitious concern ('question pneumothorax'). The original reference report is in quotes above. You can play around with your own de-identified images in this interactive web application hosted graciously by huggingface spaces.

![](./resources/streamlit.png)  

This is a super simplified example meant to demonstrate one concept. Conditionally generated radiology reports should include text inputs alongside the medical images. There are countless things to improve.

## Data
All data were derived from MIMIC and require signing a data use agreement with Physionet.  None are provided here.

## Services

This repository exposes four components that are useful in a data science proof of concept.
- A container running Jupyter notebooks with common machine learning libraries (available @ localhost:8888).  Any notebooks will persist in a mounted volume (./volumes/notebooks)
- A container running Streamlit allows a user to access the predictions from the model based on user inputs (available at localhost:8501)

## Usage

turn on the application 
```
docker-compose up 
```

download the data from physionet, passing any argument downloads the data (no arguments does nothing)
```
docker-compose run physionet True 
```

run the etl migrations
```
docker-compose run etl 
```

train the model
```
docker-compose run train
```

## Structure

```
|-- containers - code
|   |-- etl         # transforms raw data from physionet into jsonlines files
|   |-- jupyter     # interactive notebooks
|   |-- physionet   # download the MIMIC-CXR and MIMIC-CXR-JPG data from physionet
|   |-- prerad      # a small streamlit application to demo the model functionality 
|-- volumes         # persistent data
|   |-- notebooks   # jupyter notebooks persisted here
|   |-- physionet   # physionet data is persisted here
```

## References

- Chen, Zhihong, et al. "Cross-modal memory networks for radiology report generation." arXiv preprint arXiv:2204.13258 (2022).  
- Jing, Baoyu, Pengtao Xie, and Eric Xing. "On the automatic generation of medical imaging reports." arXiv preprint arXiv:1711.08195 (2017).  
- Li, Junnan, et al. "Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation." International Conference on Machine Learning. PMLR, 2022.  
- Liu, Fenglin, et al. "Exploring and distilling posterior and prior knowledge for radiology report generation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.  
- Miura, Yasuhide, et al. "Improving factual completeness and consistency of image-to-text radiology report generation." arXiv preprint arXiv:2010.10042 (2020).  
- Sirshar, Mehreen, et al. "Attention based automated radiology report generation using CNN and LSTM." Plos one 17.1 (2022): e0262209.  
- Yang, Shuxin, et al. "Knowledge matters: Chest radiology report generation with general and specific knowledge." Medical Image Analysis 80 (2022): 102510.  
- Yuan, Jianbo, et al. "Automatic radiology report generation based on multi-view image fusion and medical concept enrichment." Medical Image Computing and Computer Assisted Intervention–MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part VI 22. Springer International Publishing, 2019.   