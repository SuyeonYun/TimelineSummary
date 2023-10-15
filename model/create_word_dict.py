from collections import Counter

# 텍스트 파일을 읽어옴
with open('ocr_train.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# 문장을 단어 단위로 분리
words = text.split()

# 단어 빈도 계산
word_freq = Counter(words)

# 5회 이상 나온 단어들을 담을 딕셔너리 생성
selected_words = {word: count for word, count in word_freq.items() }

with open('selected_words_train.txt', 'w', encoding='utf-8') as file:
    for word in selected_words:
        file.write(word + '\n')