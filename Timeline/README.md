## Timeline.py
### Introduction
타임라인 생성에 가장 주가 되는 파일입니다.

해당 파일을 실행시켜 타임라인을 만들기 위한 로컬 서버를 만들 수 있습니다.

저희는 세세한 타임라인과 기본 타임라인 총 2가지 종류의 타임라인을 만들었습니다.

### Requirements
* os
* sqlite3
* pyngrok
* flask==2.3.3
* flask_cors==4.0.0

### Operating sequence
1. 요청으로부터 videoID를 가져옵니다.
2. 요청받은 영상에 대해 기존에 만들어둔 타임라인이 있는지 Timeline.db를 확인합니다.

	2-1. 타임라인이 있는 경우

		타임라인 문장을 응답으로 보내고 모든 sequence를 종료합니다.

	2-2. 타임라인이 없는 경우

		3번 sequence부터 실행합니다.

3. 영상을 다운로드 받고 1초마다 영상을 분절해 이미지를 만듭니다.
4. Craft를 실행합니다.

		Craft에 대한 내용은 './naverCraft/README.md'파일을 참고해주십시오.

5. 이미지에 있는 자막 부분만 잘라 저장합니다.
6. text recognition/text classification 모델을 실행합니다.
7. 타임라인을 추출하고 이를 Timeline.db에 저장합니다.
8. 타임라인 문장을 응답으로 보냅니다.
9. 타임라인을 만드는 동안 다운받았던 비디오, 이미지들을 삭제 후 모든 sequence를 종료합니다.



## Timeline_utils.py
### Introduction
타임라인을 만들 때 사용하는 함수들의 집합 파일입니다.

### Requirements
* os
* re
* pickle
* pytube==15.0.0
* cv2==4.8.0


### Functions
* downloadVideo_makeFrames
	* './video' 폴더에 영상을 다운로드 받고 './images' 폴더에 영상을 1초마다 분절한 이미지를 저장합니다.
* makeSubtitlesFrames
	* 자막 부분을 잘라서 './result_images' 폴더에 이미지로 저장합니다.
* makeTime
	* 초단위로 된 시간을 'mm:ss'의 형태로 바꿉니다.


## executeModel.py
### Introduction
타임라인을 만들기 위한 모델을 실행하는 파일입니다.

만들어진 타임라인은 Timeline.db 파일에 저장합니다.

### Requirements
* os
* re
* pickle
* sqlite3
* argparse
* cv2==4.8.0
* numba==0.57.1
* torch==1.13.1+cu116
* transformers==4.32.0

### Arguments
* `--videoID` : videoID
* `--treshold` : 의문형 문장에 대한 임계값
* `--flag` : 타임라인의 종류 // 0: 기본, 1: 자세히
* `--subtitlesFolder` : 자막 이미지 폴더 위치


## Timeline.db
### Introduction
타임라인을 저장하기 위한 db파일입니다.

### Tables
video, timeline

* video (예시)

pk|id|flag
:--:|:--:|:--:|
1|-Q3KrjhHE2o|0
2|-Q3KrjhHE2o|1
...|...|...|

flag는 기본 타임라인과 자세한 타임라인을 구분합니다. 0은 기본 타임라인을 의미하며 1은 자세한 타임라인을 의미합니다.

* timeline (예시)

pk|timeline|fk
:--:|:-------:|:--:|
1|0:21 연구하시는 현재 분야는 지리학 과 어떤 관련이 있나요?|1
2|0:45 과거 기후 변화가 식생 어떤 변화를 일으켰는지 연구|1
...|...|...|
14|0:21 연구하시는 현재 분야는 지리학 과 어떤 관련이 있나요?|2
15|0:45 과거 기후 변화가 식생 어떤 변화를 일으켰는지 연구|2
...|...|...|

fk는 video테이블의 참조키로 video에 대한 pk값을 가집니다.



