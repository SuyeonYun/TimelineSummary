import os
import re
import cv2
import sqlite3
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TextClassificationPipeline
from transformers import VisionEncoderDecoderModel,AutoTokenizer, TrOCRProcessor
import argparse
import pickle
from torch.utils.data import DataLoader
from numba import cuda

from Timeline_utils import makeTime

# 변수 설정
parser = argparse.ArgumentParser(description='Text Recognition')
parser.add_argument('--videoID', type=str, help='videoID')
parser.add_argument('--treshold', default=0.95, type=float, help='treshold')
parser.add_argument('--flag', default=0, type=int, help='flag')
parser.add_argument('--subtitlesFolder', default='./result_images', type=str, help='subtitlesFolder')
args = parser.parse_args()

# lengths 불러오기
with open("lengths.pkl","rb") as f:
    lengths = pickle.load(f)

# times 불러오기
with open("times.pkl","rb") as f:
    times = pickle.load(f)

# GPU 메모리 초기화
device = cuda.get_current_device()
device.reset()    

# Text Clssification 모델 불러오기
HUGGINGFACE_MODEL_PATH = "bespin-global/klue-roberta-small-3i4k-intent-classification"
loaded_tokenizer = RobertaTokenizerFast.from_pretrained(HUGGINGFACE_MODEL_PATH)
loaded_model = RobertaForSequenceClassification.from_pretrained(HUGGINGFACE_MODEL_PATH)

text_classifier = TextClassificationPipeline(
    tokenizer=loaded_tokenizer,
    model=loaded_model,
    top_k=1
)

# TrOCR 모델 불러오기
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
device = torch.device('cuda')
trocr_model = "gg4ever/trOCR-final"
model = VisionEncoderDecoderModel.from_pretrained(trocr_model).to(device)
tokenizer = AutoTokenizer.from_pretrained(trocr_model)

# DB 불러오기
con = sqlite3.connect('timeline.db')
cur = con.cursor()

# 자막 부분 이미지 정렬해서 불러오기
subtitles = sorted(os.listdir(args.subtitlesFolder), key = lambda x: (int(re.search(r'(\d{1,})',x).group(1)),\
                                                           int(re.search(r'_(\d{1,})',x).group(1))))

# DB에 videoID 저장
cur.execute('INSERT INTO video(id, flag) VALUES(?,?)',(args.videoID, args.flag))

# 모델 실행할 때 필요한 변수 설정
subtitleTemp = ''
subtitle = ''
th = args.treshold
i = 0
j = len(lengths)

# 모델 실행
while lengths:
    time = times.pop(0)
    temp = []

    print("Checking subtitles in frame {:d}/{:d}".format(i,j), end='\r')
    i += 1
    datas = []
    
    # 자막 이미지 불러오기
    for _ in range(lengths.pop(0)):
        subtitlePath = args.subtitlesFolder + '/' + subtitles.pop(0)
        datas.append(cv2.resize(cv2.imread(subtitlePath),dsize=(128,64)))
    
    if not datas:
        continue
        
    # data_loader 생성
    data_loader = DataLoader(datas, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
    data_loader = iter(data_loader)
    
    # TrOCR 모델 실행
    for _ in range(len(data_loader)):
        data = next(data_loader)
        pixel_values = (processor(data, return_tensors="pt").pixel_values).to(device)
        temp = temp + tokenizer.batch_decode(model.generate(pixel_values, max_length=32), skip_special_tokens=True)
        
    # 한국어만 추출
    temp = ' '.join(re.findall(r'[가-힣?!]+', ' '.join(temp)))
    
    # Text Classification 모델 실행
    preds_list = text_classifier(temp)
    pred = sorted(preds_list[0], key = lambda x: x['score'], reverse = True)[0]
    
    # 읽어온 자막이 의문형인지 확인
    if pred['label'] != 'question':
        continue
        
    # 읽어온 자막이 의문형이고 설정한 임계치를 넘는지 확인
    elif pred['score'] > th:
        subtitleTemp = subtitle
        subtitle = temp
        
        # 앞선 자막과 중복 제거
        if len(set(subtitle) & set(subtitleTemp)) > round(min(len(set(subtitleTemp)),len(set(subtitle)))*0.55):
            continue
            
        else:
            # 초단위로 된 시간을 mm:ss 형태로 변경
            timeLine = makeTime(time)
            
            # DB에 타임라인 저장
            cur.execute('INSERT INTO timeline(fk, timeline) VALUES((SELECT pk FROM video WHERE id = ? AND flag = ?),?)',\
                       (args.videoID, args.flag, f'{timeLine} {subtitle}'))

# DB 저장 및 닫기
con.commit()
con.close()

