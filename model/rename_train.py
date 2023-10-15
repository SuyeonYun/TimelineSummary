import os

labels_file_list = ["labels_background.txt", "labels_basic.txt", "labels_blur_1.txt", "labels_blur_2.txt", "labels_distortion.txt"]


image_folder_list = ["train_background","train_basic","train_blur_1","train_blur_2","train_distortion"]


no_image_dict = {}

# 레이블 파일 열기
for labels_file_path, image_folder in zip(labels_file_list,image_folder_list):
    with open(labels_file_path, 'r', encoding='utf-8') as labels_file:
        lines = labels_file.readlines()
# 라벨 정보를 반복하며 이미지 파일 이름 변경
    for line in lines:
        line_parts = line.split(" ")
        image_number = line_parts[0][:-4]  # 이미지 번호 추출 (숫자).jpg에서 .jpg 제거
        new_image_name = f"{image_folder[6:]}_{image_number}.jpg"  # 새로운 이미지 파일 이름 train_ 제거.
        original_image_path = os.path.join(image_folder, line_parts[0])
        new_image_path = os.path.join(image_folder, new_image_name)
        # 이미지 파일 이름 변경
        try:
            os.rename(original_image_path, new_image_path)
        except:
            #image가 없는 오류 파일을 dictionary에 추가
            no_image_dict[new_image_name]=1

print("이미지 파일 이름 변경 완료!")

new_files_list = ["labels_background_renamed.txt", "labels_basic_renamed.txt", "labels_blur_1_renamed.txt", "labels_blur_2_renamed.txt", "labels_distortion_renamed.txt"]

for (labels_file_path,new_labels_file_path) in zip(labels_file_list,new_files_list):
    # 레이블 파일 열기
    with open(labels_file_path, 'r', encoding='utf-8') as labels_file:
        lines = labels_file.readlines()

    # 새로운 레이블 파일 생성 및 기존 레이블 정보를 반복하며 변환하여 저장
    with open(new_labels_file_path, 'w', encoding='utf-8') as new_labels_file:
        for line in lines:
            line_parts = line.split(" ")
            image_number = line_parts[0][:-4]  # 이미지 번호 추출 (숫자).jpg에서 .jpg 제거
            new_image_name = f"{labels_file_path[7:-4]}_{image_number}.jpg"  # 새로운 이미지 파일 이름
            if new_image_name in no_image_dict:
                #이미지가 없으면 labels에 적지 않는다.
                continue
            new_line = f"word1/{new_image_name} {' '.join(line_parts[1:])}"  # 새로운 라인 생성
            new_labels_file.write(new_line)

print("레이블 파일 변환 및 저장 완료!")