# 데이터 전처리 , 모델 학습

# 1. TextRecognitionDataGenerator로 구어체 단어 imageset 생성.

## (1) Text data 수집 ->text_data.zip

- AIhub 주제별 텍스트 일상 대화 : https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=543
- 챗봇 데이터셋(Q,A)로 구성 중 Q 부분 : https://github.com/songys/Chatbot_data

-> pandas로 합쳐서 ocr_train.txt로 저장

-> create_word_dict.py로 단어를 추출해서 selected_words_train.txt 로 저장

-> 여기서 나온 단어들을 trdg의 dict 폴더에 저장.

## (2) Image 생성

-> word 1개 생성.txt로 trdg/run.py를 실행시켜 dict안의 단어 중 무작위로 선정하여 background(뒷배경 있는), blur_1(적은 blur), blur_2(강한 blur), basic(뒷배경 없음), distortion(왜곡) 에 글씨 이미지를 생성하고 저장. 이때 생성되는 labels.txt는 (093928,'안녕') 식으로 저장되어 있어 폴더 구분이 필요

-> labels.txt를 (index,filename,text) 형식으로 만들어주는게 rename_train.py

-> 이때 이 4개의 폴더를 합쳐서 word1 폴더에 저장하는 것이 concat_train_image.py

-> 이것으로 생성된 train_background_renamed.txt ... 를 합쳐주는 것이 make_train_df.py

-> 이것으로 word1.txt 생성 (run1.py만 돌리면 word1.txt를 생성하게끔 만들어뒀습니다.)

# 2. Aihub 단어 OCR 데이터 가공

- https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=81

-> 단어 인쇄체 2234400.png~2497152.png ->약 26만장
-> 단어 augmentation 2511550.png~3342999.png -> 약 84만장
-> 총 110만장
-> json에서 annotation->text를 추출해서 .png파일과 (filename,text)가 담긴 word2.txt 파일을 제출하고 싶었으나 png폴더크기가 너무 커서 word2.txt 파일만 저장해서 올림 png 파일은 aihub에서 다운로드받을 수 있고, word2.txt 만드는 코드는 move_aihub_image.py에 있고, word2.txt 만드는법은 run2.ipynb에 저장

# 3. 최종 전처리 결과

1. word1(100만장)
   --background_0.jpg
   --background_1.jpg
   ...
   --distortion_199999.jpg

2. word2(약 110만장)
   --02234400.png
   --02234401.png
   ...
   --03342999.png

3. annotations
   --word1.txt
   --word2.txt

# 4.모델 학습

## (1) image2text model

- text2image_model.py
- 모델은 trOCR을 이용, hugging face를 이용해서 VisionEncoderDecoder(Encoder : pre-trained DeiT, Decoder+tokenizer : pre-trained klue-RoBERTa)

- 결과 모델은 hugging face에 저장함.
- https://huggingface.co/gg4ever/trOCR-final

## (2) text type classification model

- text_classification_model.py
- klue-RoBERTa 를 finetuning해서 만듦.

- 결과 모델은 hugging face에 저장함.
- https://huggingface.co/gg4ever/intent-classification-korean
