from datasets import load_dataset
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from transformers import TrainingArguments, Trainer
import numpy as np
from copy import deepcopy
from transformers import TrainerCallback
from typing import *
from sklearn.metrics import accuracy_score
import time
from transformers.trainer_utils import speed_metrics
import math
import matplotlib.pyplot as plt
import torch

# BERT dataset 설정
class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, pair_dataset, label):
        self.pair_dataset = pair_dataset
        self.label = label

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['label'] = torch.tensor(self.label[idx])

        return item

    def __len__(self):
        return len(self.label)

# sklearn의 accuracy score 도입
def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds) 

  return {
      'accuracy': acc,
  }

# CTC trainer는 train_accuracy를 hugging face에서 지원을 하지 않기에 임의로 Trainer기능을 가진 CTCTrainer을 만들어서, eval_steps마다 train_accuracy를 측정하기 위한 것입니다.
# image2text모델에서는 train_accuracy를 구하면 너무너무 학습시간이 길어져서 사용하지 않았습니다.
class CTCTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        train_dataloader = self.get_train_dataloader()
        start_time = time.time()
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop

        eval_output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        train_output = eval_loop(
            train_dataloader,
            description='Training Evaluation',
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix="train",
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in eval_output.metrics:
            start_time += eval_output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        eval_output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=eval_output.num_samples,
                num_steps=math.ceil(eval_output.num_samples / total_batch_size),
            )
        )
        train_n_samples = len(self.train_dataset)
        train_output.metrics.update(speed_metrics('train', start_time, train_n_samples))
        self.log(train_output.metrics | eval_output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, train_output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, eval_output.metrics)
        self._memory_tracker.stop_and_update_metrics(eval_output.metrics)
        self._memory_tracker.stop_and_update_metrics(train_output.metrics)

        return train_output.metrics | eval_output.metrics

#데이터셋 hugging face로부터 불러오기.
dataset = load_dataset("kor_3i4k")

#데이터셋은 train과 test가 10:1로 나뉘어져 있음
train_texts = dataset['train']["text"]
test_texts = dataset['test']["text"]
train_labels = dataset['train']["label"]
test_labels = dataset['test']["label"]

#val_data 를 생성해야하기 때문에, train을 한번 더 쪼개서 train:val:test를 9:1:1로 만들어줌.
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1, random_state=0)

train_data = {
    "label": train_labels,
    "text": train_texts
}
train_dataset = Dataset.from_dict(train_data)

val_data = {
    "label": val_labels,
    "text": val_texts
}
val_dataset = Dataset.from_dict(val_data)

test_data = {
    "label": test_labels,
    "text": test_texts
}
test_dataset = Dataset.from_dict(test_data)

#pre-trained 모델 가져오기 : RoBERTa
MODEL_NAME = 'klue/roberta-base'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# config를 3i4k label에 맞게 변경해주기.
config = AutoConfig.from_pretrained(MODEL_NAME)
config.num_labels = 7
config.id2label={
    0: "fragment",
    1: "statement",
    2: "question",
    3: "command",
    4: "rhetorical question",
    5: "rhetorical command",
    6: "intonation-dependent utterance"
}
config.label2id={
    "fragment" : 0,
    "statement" : 1,
    "question" : 2,
    "command" : 3,
    "rhetorical question" : 4,
    "rhetorical command" : 5,
    "intonation-dependent utterance" : 6
}

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

# RoBERTa tokenizer에 맞게 데이터셋변경
tokenized_train = tokenizer(
    list(train_dataset['text']),
    return_tensors="pt",
    max_length=256, # Max_Length = 190
    padding=True,
    truncation=True,
    add_special_tokens=True
)

tokenized_eval = tokenizer(
    list(val_dataset['text']),
    return_tensors="pt",
    max_length=256,
    padding=True,
    truncation=True,
    add_special_tokens=True
)

tokenized_test = tokenizer(
    list(test_dataset['text']),
    return_tensors="pt",
    max_length=256,
    padding=True,
    truncation=True,
    add_special_tokens=True
)

train_dataset = BERTDataset(tokenized_train, train_dataset['label'])
val_dataset = BERTDataset(tokenized_eval, val_dataset['label'])
test_dataset = BERTDataset(tokenized_test, test_dataset['label'])

# hyperparameter 설정
training_args = TrainingArguments(
    evaluation_strategy="steps",
    output_dir="test_trainer",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    learning_rate=4e-5,
    logging_steps=400,
    eval_steps=400,
    save_steps=2000,
    warmup_steps=1000,
)

# trainer 선언
trainer = CTCTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 훈련 시작
trainer.train()

# 모델 저장
trainer.save_model(output_dir="./model")

# test dataset을 evaluation 하기
model.eval()
with torch.no_grad():
    eval_result = trainer.evaluate(test_dataset)
print(eval_result)

# train_loss, eval_loss 그래프 그리기
if True:
    train_steps = []
    eval_steps=[]
    train_losses = []
    eval_losses=[]
    for obj in trainer.state.log_history:
        if obj['step']%200 == 0:
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


''' #train accuracy, eval_accuracy 그래프 그리기 
import matplotlib.pyplot as plt
if True:
    eval_steps=[]
    train_acc=[]
    eval_acc=[]
    for obj in trainer.state.log_history:
        print(obj)
        if 'train_accuracy' in obj:
            eval_steps.append(obj['step'])
            train_acc.append(obj['train_accuracy'])
            eval_acc.append(obj['eval_accuracy'])

f = plt.figure(figsize=(12,6))
plt.plot(eval_steps, train_acc, label='train_acc', color='violet')
plt.plot(eval_steps, eval_acc, label='eval_acc', color='limegreen')
plt.xlabel('step')
plt.ylabel('accuracy')
plt.legend()
plt.show()
'''