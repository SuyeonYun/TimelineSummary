from flask import Flask
from flask_cors import CORS
from flask import request
from pyngrok import conf, ngrok
import os
import sqlite3

from Timeline_utils import downloadVideo_makeFrames, makeSubtitlesFrames

# API 생성
api = Flask(import_name='__name__')

# CORS 허용
CORS(api, origins="https://www.youtube.com") 

conf.get_default().region = "jp"
http_tunnel = ngrok.connect(5000)
tunnels = ngrok.get_tunnels()

# tunnels 출력
for tunnel in tunnels:
    print(tunnel)
    
    
@api.route('/makeTL', methods = ['GET'])
def detail():
    print('We take request by detail.')
    
    # 요청 url로부터 videoID 받아오기
    videoID = request.args.get('v', None) 
    
    # DB 불러오기
    con = sqlite3.connect('timeline.db')
    con.row_factory = lambda cursor, row: row[0]
    cur = con.cursor()
    
    # DB에 기존에 만들었던 타임라인이 있는지 확인
    cur.execute('SELECT pk FROM video WHERE id = ? AND flag = 1',[videoID])
    pk = cur.fetchall()
    
    # 기존에 만든 타임라인이 있는 경우 이를 응답
    if pk:
        cur.execute('SELECT timeline FROM timeline WHERE fk = ?',[pk[0]])
        subtitles = cur.fetchall()
        con.close()
        if subtitles:
            return '<br>'.join(subtitles)
        else:
            return 'Sorry..<br>We cannot make TimeLine'
    
    # 기존에 만든 타임라인이 없는 경우 타임라인 만들기
    else:
        
        # 영상을 './video'폴더에 다운받아 1초마다 영상을 분절해 './images'폴더에 이미지 저장
        downloadVideo_makeFrames(videoID, './video', './images')
        
        # Craft 실행 및 bounding box 좌표를 './result'폴더에 txt파일로 저장
        os.system("python ./naverCraft/test.py --trained_model=./naverCraft/craft_mlt_25k.pth --test_folder=./images --text_threshold=0.8")

        # './result'폴더 내 bounding box 좌표에 따라 './images'폴더에 있는 이미지에서 자막 부분만 잘라서 './result_images'폴더에 저장
        makeSubtitlesFrames('./result', './images', './result_images')
        
        # 세세한 타임라인을 만들기 위한 모델 실행 및 DB에 저장
        os.system(f'python executeModel.py --videoID={videoID} --treshold=0.85 --flag=1')
        
        # DB에 저장된 타임라인을 검색
        cur.execute('SELECT pk FROM video WHERE id = ? AND flag = 1',[videoID])
        pk = cur.fetchall()
        cur.execute('SELECT timeline FROM timeline WHERE fk = ?',[pk[0]])
        subtitle = cur.fetchall()
        con.commit()
        con.close()
        
        # 타임라인을 만들 때 저장했던 영상, 이미지들 삭제
        os.remove('./lengths.pkl')
        os.remove('./times.pkl')
        
        imgs = os.listdir('./images')
        while imgs:
            os.remove('./images/' + imgs.pop(0))
            
        os.remove('./video/' + os.listdir('./video')[0])
        
        infos = os.listdir('./result')
        while infos:
            os.remove('./result/' + infos.pop(0))
            
        images = os.listdir('./result_images')
        while images:
            os.remove('./result_images/' + images.pop(0))
            
        # 타임라인 응답
        if subtitle:
            return '<br>'.join(subtitle)
        else:
            return 'Sorry . . We cannot make Timeline'

        
        
@api.route('/makemore', methods = ['GET'])
def basic():
    print('We take request by basic.')
    
    # 요청 url로부터 videoID 받아오기
    videoID = request.args.get('v', None) 
    
    # DB 불러오기
    con = sqlite3.connect('timeline.db')
    con.row_factory = lambda cursor, row: row[0]
    cur = con.cursor()
    
    # DB에 기존에 만들었던 타임라인이 있는지 확인
    cur.execute('SELECT pk FROM video WHERE id = ? AND flag = 0',[videoID])
    pk = cur.fetchall()
    
    # 기존에 만든 타임라인이 있는 경우 이를 응답
    if pk:
        cur.execute('SELECT timeline FROM timeline WHERE fk = ?',[pk[0]])
        subtitles = cur.fetchall()
        con.close()
        if subtitles:
            return '<br>'.join(subtitles)
        else:
            return 'Sorry..<br>We cannot make TimeLine'
    
    # 기존에 만든 타임라인이 없는 경우 타임라인 만들기
    else:
        
        # 영상을 './video'폴더에 다운받아 1초마다 영상을 분절해 './images'폴더에 이미지 저장
        downloadVideo_makeFrames(videoID, './video', './images')
        
        # Craft 실행 및 bounding box 좌표를 './result'폴더에 txt파일로 저장
        os.system("python ./naverCraft/test.py --trained_model=./naverCraft/craft_mlt_25k.pth --test_folder=./images --text_threshold=0.8")

        # './result'폴더 내 bounding box 좌표에 따라 './images'폴더에 있는 이미지에서 자막 부분만 잘라서 './result_images'폴더에 저장
        makeSubtitlesFrames('./result', './images', './result_images')
        
        # 타임라인을 만들기 위한 모델 실행 및 DB에 저장
        os.system(f'python executeModel.py --videoID={videoID}')
        
        # DB에 저장된 타임라인을 검색
        cur.execute('SELECT pk FROM video WHERE id = ? AND flag = 0',[videoID])
        pk = cur.fetchall()
        cur.execute('SELECT timeline FROM timeline WHERE fk = ?',[pk[0]])
        subtitle = cur.fetchall()
        con.commit()
        con.close()
        
        # 타임라인을 만들 때 저장했던 영상, 이미지들 삭제
        os.remove('./lengths.pkl')
        os.remove('./times.pkl')
        
        imgs = os.listdir('./images')
        while imgs:
            os.remove('./images/' + imgs.pop(0))
            
        os.remove('./video/' + os.listdir('./video')[0])
        
        infos = os.listdir('./result')
        while infos:
            os.remove('./result/' + infos.pop(0))
            
        images = os.listdir('./result_images')
        while images:
            os.remove('./result_images/' + images.pop(0))
            
        # 타임라인 응답
        if subtitle:
            return '<br>'.join(subtitle)
        else:
            return 'Sorry . . We cannot make Timeline'
    
    
    
if __name__ == '__main__':
    api.run(debug = True, threaded=False)

