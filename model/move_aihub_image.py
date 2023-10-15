import os
from tqdm import tqdm

source_folders = ['02.인쇄체_230721_add/01_printed_word_images/word','03.인쇄체 증강데이터/zip4','03.인쇄체 증강데이터/zip5', '03.인쇄체 증강데이터/13-6']

# 대상 폴더
destination_folder = 'word2'

# 대상 폴더가 없다면 생성
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

dest_path=os.path.join(os.getcwd(),destination_folder)

for src in source_folders:
    source_path=os.path.join(os.getcwd(), src)
    files = os.listdir(source_path)
    for f in tqdm(files, desc=f"Moving files from {src}"):
        if f.endswith(".png"):
            src_file_path = os.path.join(source_path, f)
            dest_file_path = os.path.join(dest_path, f)
            os.rename(src_file_path, dest_file_path)

print("이미지 파일들을 word2에 성공적으로 저장했습니다.")