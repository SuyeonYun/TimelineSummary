from pytube import YouTube
import cv2
import pickle
import os
import re

# 영상을 다운받고 영상을 초당 이미지로 분절하는 함수
def downloadVideo_makeFrames(videoID, videoPath, framesPath):
    
    # './videoPath' 폴더에 영상 다운로드
    url = "https://www.youtube.com/watch?v=" + videoID
    yt = YouTube(url)
    stream = yt.streams.get_highest_resolution()
    print('Downloading video . . .')
    stream.download(videoPath)
    
    filepath = os.listdir(videoPath)[0]
    filepath = videoPath + '/' + filepath
    
    # 영상 정보 설정
    video = cv2.VideoCapture(filepath)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(video.get(cv2.CAP_PROP_FPS))

    # 초당 이미지로 영상 분절 후 './framesPath' 폴더에 저장
    print('Making frames . . .')
    count = 0
    while(video.isOpened()):
        if video.get(1) >= length:
            break
        ret, image = video.read()
        if(int(video.get(1)) % fps == 0):
            cv2.imwrite(framesPath + "/frame%d.jpg" % count, image)
            count += 1
    video.release()
    
# 자막 부분만 잘라서 이미지 저장
def makeSubtitlesFrames(infoPath, imagesPath, resultPath):
    print('Making subtitle Frames . . .')
    
    # './infoPath' 폴더에 있는 bounding box 정보 가져오기
    infos = sorted(os.listdir(infoPath), key = lambda x: int(re.search(r'(\d{1,})',x).group()))

    lengths = []
    times = []
    
    # 자막 이미지 만들기
    for i in range(len(infos)):
        with open(infoPath + '/' + infos[i], "r") as data:
            info = data.read()
            if info:
                points = re.sub('\n\n',',',info.strip()).split(',')
                
                # 자막 개수 정보 저장
                lengths.append(int(len(points)/8))
                
                # 이미지 시간 정보 저장
                times.append(i + 1)
                
                widthPoints = []
                heightPoints = []

                while points:
                    widthTemp = []
                    heightTemp = []
                    for _ in range(4):
                        widthTemp.append(int(points.pop(0)))
                        heightTemp.append(int(points.pop(0)))
                        
                    # Bounding box 좌표 불러오기
                    widthPoints.append(widthTemp)
                    heightPoints.append(heightTemp)

                
                for k in range(len(widthPoints)):
                    # 이미지 height가 30보다 낮은 경우 제외
                    if max(heightPoints[k]) - min(heightPoints[k]) < 30:
                        lengths[-1] = lengths[-1] - 1
                        continue
                        
                    # './imagesPath' 폴더 내 이미지를 불러와 자막부분만 자르고 './resultPath' 폴더에 저장
                    img = cv2.imread(imagesPath + f'/frame{i}.jpg')
                    img = img[int(abs(min(heightPoints[k]))):int(abs(max(heightPoints[k]))), int(abs(min(widthPoints[k]))):int(abs(max(widthPoints[k]))), :]
                    cv2.imwrite(resultPath + f'/frame{i}_{k}.jpg',img)
                print("Making Subtitle Frame {:d}/{:d}".format(i+1, len(infos)), end='\r')
    print('Make All Subtitle Frames!!')
    
    # lengths 저장
    with open("lengths.pkl","wb") as f:
        pickle.dump(lengths, f)
    
    # times 저장
    with open("times.pkl","wb") as f:
        pickle.dump(times, f)
        

# 초단위 시간을 mm:ss 형태로 변환
def makeTime(time):
    if time > 3600:
        if time % 60 < 10:
            timeLine = f'{time // 3600}:{(time // 3600) // 60}:0{time % 60}'
        else:
            timeLine = f'{time // 3600}:{(time // 3600) // 60}:{time % 60}'

    else:
        if time % 60 < 10:
            timeLine = f'{time // 60}:0{time % 60}'
        else:
            timeLine = f'{time // 60}:{time % 60}'
    
    
    return timeLine

