'''
    메인 서비스를 구축하는 컨트롤러
    - 라우트 : URL과 이를 처리할 함수 연계
    - 비즈니스 로직 : 사용자가 요청하는 주 내용을 처리하는 곳
'''
from flask import Flask, render_template, request, make_response, url_for, jsonify, Response
from  service.controllers   import bp_auth as auth
# 시간정보획득, 시간차를 계산하는 함수
from datetime import datetime, timedelta
import time
# Flask 객체 획득
from flask import current_app
import jwt

import bcrypt

from service.model.models import Information
from datetime import datetime                
from service import db

import boto3

# 플라스크 객체뿐 아니라 등록한 blueprint 객체도 라우팅 가능
# ~/auth/
# @app.route('/', methods=['GET', 'POST'])
@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        # jwt 관련 체크 => 정상(200), 오류(404)
        # 1.uid, upw 획득
        uid = request.form.get('uid')
        upw = request.form.get('upw')
        info = Information.query.filter(Information.uid==uid).all() 

        # 2.uid, upw로 회원이 존재하는지 체크 
        if info and info[0].uid == uid and info[0].upw:
            # 비밀번호 확인 checkpw(그냥 바이트값, 해쉬값) -> bool
            checkpw = bcrypt.checkpw(upw.encode('utf-8'), info[0].upw.encode('utf-8'))
            if not checkpw :
                return jsonify({'msg': "입력하신 정보가 없습니다. 다시 시도해주세요!"})
            # 3.회원이면 토큰 생성 (규격, 만료시간, 암호 알고리즘 지정, ...)
            payload = { 
                # 저장할 정보는 알아서 구성(고객 정보가 기반)
                'id': info[0].id, 
                # 만료시간 (원하는대로 설정)
                # 토큰이 발급되고 나서 + 24시간 후에 토큰은 만료된다
                'exp': datetime.utcnow() + timedelta(seconds=60*60*24) 
            }
            # 토큰 발급 => 시크릿키, 해시알고리즘("HS256"), 데이터(payload)
            SECRET_KEY = current_app.config['SECRET_KEY']   # 환경변수값 획득
            # 발급
            token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
            # # 4. 응답 전문 구성 -> 응답
            return jsonify({'code':1, 'token':token})
        else :
            return jsonify({'msg': "입력하신 정보가 없습니다. 다시 시도해주세요."})

# @auth.route('/logout')
# def logout():
#     return "auth logout"
@auth.route('/loading')
def loading():
    return render_template('loading.html')

@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('signup.html')
    else:
        # jwt 관련 체크 => 정상(200), 오류(404)
        # 1.uid, upw 획득
        uid = request.form.get('uid')
        upw = request.form.get('upw')
        cname = request.form.get('cname')
        caccent = request.form.get('caccent')
        
        # 아이디 중복확인
        info = Information.query.filter(Information.uid==uid).all() 
        if info:
            return jsonify({'msg' : '아이디가 중복됩니다. 새로 입력해주세요.'})

        bupw = (bcrypt.hashpw(upw.encode('utf-8'), bcrypt.gensalt())).decode('utf-8')

        a = Information(uid=uid, upw=bupw, cname=cname, caccent=caccent, reg_date=datetime.now(), up_date=datetime.now()) 
        # 신규 가입 등록
        db.session.add(a)
        # 커밋
        db.session.commit()

        info = Information.query.filter(Information.uid==uid).all() 
        
        if info and info[0].id:
            # 이미지 업로드
            s3 = boto3.resource('s3')
            print( dir(request.files), request.files.get('file') ) 
            file = request.files['file']
            print(file)
            user_id = info[0].id
            s3.Bucket('ai-project4-group6').put_object(Key='GANS/datas/' + str(user_id) + '/' + file.filename, Body=file)
            print('업로드 완료')
            return jsonify({'id' : info[0].id})
        else:
            return jsonify({'msg' : '오류가 발생했습니다. 다시 시도해주세요.'})

# @auth.route('/delete')
# def delete():
#     return "auth delete"

    