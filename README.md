# A new generative model for radiology

tl;dr you should include text inputs along with images

Machine learning in radiology has come a long way. For a long time the goal was simply to make a probability estimates of different conditions available to the radiologist at the time of interpretation. As evidence, see any of the hundreds of AI vendors that have commercialized computer vision algorithms. On the academic frontier, recent advances have made it possible to generate realistic sounding radiology reports directly from an image. The first paper I found describing such a model was from 2017, but there have been many more recently with the onset of transformers. However, every example I have found suffers from the same structural problem. 

__They aren't answering a clinical question.__

In my literature search I found seven studies describing model architectures that could conditionally generate entire radiology reports from an image, but none had text inputs (Jing et al. 2017, Yuan et al. 2019, Miura et al. 2020, Fenglin et al. 2021, Sirshar et al. 2022, Chen et al. 2022, Yang et al. 2022).  This is a problem because all the variation in the generated reports will come from the images (thus, not answering a clinical question in posed in text by the ordering provider).

To realistically describe what a radiologist is doing when they write a report, we need to utilize the same inputs. This means the conditional generation of radiology reports should include both image and text inputs and have a text output. This year a new transformer architecture particularly suited for this type of multi-modal problem was just released by SalesForce (Li et al. 2022). BLIP has a dual text and vision encoder paired with a text decoder. This allows it to continue generating new text for a radiology report from a given prompt's starting point. Lucky for us, the first paragraph of most radiology reports is the clinical question!

Starting from the base BLIP image captioning model, I fine-tuned a causal language model to generate radiology reports from a chest x-ray and a small prompt. 

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

## Streamlit User Interface

![](./resources/streamlit.png)  

A small web application can take a clinical indication and a chest x-ray, returns a reasonably templated radiology report.

## References

- Chen, Zhihong, et al. "Cross-modal memory networks for radiology report generation." arXiv preprint arXiv:2204.13258 (2022).  
- Jing, Baoyu, Pengtao Xie, and Eric Xing. "On the automatic generation of medical imaging reports." arXiv preprint arXiv:1711.08195 (2017).  
- Li, Junnan, et al. "Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation." International Conference on Machine Learning. PMLR, 2022.  
- Liu, Fenglin, et al. "Exploring and distilling posterior and prior knowledge for radiology report generation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.  
- Miura, Yasuhide, et al. "Improving factual completeness and consistency of image-to-text radiology report generation." arXiv preprint arXiv:2010.10042 (2020).  
- Sirshar, Mehreen, et al. "Attention based automated radiology report generation using CNN and LSTM." Plos one 17.1 (2022): e0262209.  
- Yang, Shuxin, et al. "Knowledge matters: Chest radiology report generation with general and specific knowledge." Medical Image Analysis 80 (2022): 102510.  
- Yuan, Jianbo, et al. "Automatic radiology report generation based on multi-view image fusion and medical concept enrichment." Medical Image Computing and Computer Assisted Intervention–MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part VI 22. Springer International Publishing, 2019.   