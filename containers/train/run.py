import pandas as pd
from datasets import Dataset, Image
import torch
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import BlipProcessor, BlipForConditionalGeneration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model initialize form pretrained
repo = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(repo)
tokenizer = processor.tokenizer
model = BlipForConditionalGeneration.from_pretrained(repo)

# load the data configuration and split into test/valid
dt = pd.read_json("dataset.jsonl",lines=True).dropna()
dt["train"] = dt["fold"].apply(lambda x: 0 if x in ['p19'] else 1) # 10% of data
dt["patient"]= dt["patient"].apply(lambda x: x[0:5])
train=dt[dt.train==1]
valid=dt[dt.train==0]

# create datasets
train_dataset = Dataset.from_dict({
    "image": train["image"].to_list(),
    "fold": train["fold"].to_list(),
    "text": train["text"].to_list(),
    "reason": train["reason"].to_list(),
    "id": [x.split("/")[-1].replace(".jpg","") for x in train["image"].to_list()]
}).cast_column("image", Image())

valid_dataset = Dataset.from_dict({
    "image": valid["image"].to_list(),
    "fold": valid["fold"].to_list(),
    "text": valid["text"].to_list(),
    "reason": valid["reason"].to_list(),
    "id": [x.split("/")[-1].replace(".jpg","") for x in valid["image"].to_list()]
}).cast_column("image", Image())

def transform(example_batch):
    return processor(
        images=[image for image in example_batch["image"]],
        text=[text for text in example_batch["text"]],
        return_tensors="np",
        padding='max_length',
        max_length=512
    )

# apply 
train_prepared = train_dataset.shuffle(seed=42).with_transform(transform)
valid_prepared = valid_dataset.shuffle(seed=42).with_transform(transform)

# " ".join(processor.batch_decode(train_prepared[0]["input_ids"])).replace(" ##","")
training_args = TrainingArguments(
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_steps=1000,
    logging_steps=100,
    per_device_eval_batch_size=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    lr_scheduler_type='cosine_with_restarts',
    warmup_ratio=0.1,
    learning_rate=5e-5,
    save_total_limit=1,
    output_dir="/opt/models/generate-cxr-checkpoints"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm = False
)

trainer = Trainer(
    model=model,
    tokenizer=processor,
    args=training_args,
    train_dataset=train_prepared,
    eval_dataset=valid_prepared,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("/opt/models/generate-cxr")
