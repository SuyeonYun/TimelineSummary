import numpy as np
import pandas as pd
import torch
import evaluate
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator

class OCRDataset(Dataset):
    def __init__(self, dataset_dir, df, processor, tokenizer, max_target_length=32):
        self.dataset_dir = dataset_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.dataset_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.tokenizer(text, padding="max_length",
                                stride=32,
                                truncation=True,
                                max_length=self.max_target_length).input_ids

        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
    label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer, "wer": wer}


#word1.txt, worc2.txt를 한 데이터프레임으로 만들어주기
df_word1=pd.read_csv("annotations/word1.txt")
df_word1.columns={'file_name','text'}
df_word2=pd.read_csv("annotations/word2.txt")
df_word2.columns={'file_name','text'}
df = pd.concat(
         [df_word1, df_word2], ignore_index=True
)

# word2가 너무 커서 못담았는데, word1만 써서 하려면 다음 코드를 사용하시면됩니다.
# df_word1=pd.read_csv("annotations/word1.txt")
# df_word1.columns={'file_name','text'}
# df=df_word1

#pre-train 모델 설정 : Encoder: DeiT, Decoder: RoBERTa
vision_hf_model = 'facebook/deit-base-distilled-patch16-384' #DeiT
nlp_hf_model = "klue/roberta-base" #한국어 pre-trained RoBERTa
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(vision_hf_model, nlp_hf_model)
tokenizer = AutoTokenizer.from_pretrained(nlp_hf_model)

# train-eval dataset 분리
train_df, test_df = train_test_split(df, test_size=0.06, random_state=1234)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

#text를 토큰화시켜 데이터셋생성
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
train_dataset_dir = ''
valid_dataset_dir = ''
max_length = 32

train_dataset = OCRDataset(
    dataset_dir=train_dataset_dir,
    df=train_df,
    tokenizer=tokenizer,
    processor=processor,
    max_target_length=max_length
)
eval_dataset = OCRDataset(
    dataset_dir=valid_dataset_dir,
    df=test_df,
    tokenizer=tokenizer,
    processor=processor,
    max_target_length=max_length
)

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = tokenizer.sep_token_id
model.config.max_length = max_length
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

# hypterparameter 설정
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2, 
    fp16=True,
    learning_rate=4e-5,
    output_dir="./models",
    save_steps=40000,
    eval_steps=10000,
    warmup_steps=20000, 
    weight_decay=0.01
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)

#훈련 시작
result = trainer.train()

#models 폴더에 저장
trainer.save_model(output_dir="./model")

#train_loss, eval_loss 그래프 확인
train_steps = []
eval_steps=[]
train_losses = []
eval_losses=[]
for obj in trainer.state.log_history:
    if obj['step']%10000 == 0:
        if 'loss' in obj :
            train_steps.append(obj['step'])
            train_losses.append(obj['loss'])
        else:
            eval_steps.append(obj['step'])
            eval_losses.append(obj['eval_loss'])

f = plt.figure(figsize=(12,6))
plt.plot(train_steps, train_losses, label='train_loss')
plt.plot(eval_steps, eval_losses, label='eval_loss')
plt.xlabel('step')
plt.ylabel('loss')
plt.legend()
plt.show()