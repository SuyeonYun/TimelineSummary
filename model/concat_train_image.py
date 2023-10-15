import os
from tqdm import tqdm

source_folders = ['train_background','train_basic','train_blur_1','train_blur_2', 'train_distortion']

# 대상 폴더
destination_folder = 'word1'

# 대상 폴더가 없다면 생성
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

dest_path=os.path.join(os.getcwd(),destination_folder)

for src in source_folders:
    source_path=os.path.join(os.getcwd(), src)
    files = os.listdir(source_path)
    for f in tqdm(files, desc=f"Moving files from {src}"):
        if f.endswith(".jpg"):
            src_file_path = os.path.join(source_path, f)
            dest_file_path = os.path.join(dest_path, f)
            os.rename(src_file_path, dest_file_path)

print("이미지 파일들을 word1에 성공적으로 저장했습니다.")