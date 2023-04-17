# jwt
import jwt
# 시간 관련
import time
from datetime import datetime, timedelta

'''
    메인 서비스를 구축하는 컨트롤러
    - 라우트 : URL과 이를 처리할 함수 연계
    - 비즈니스 로직 : 사용자가 요청하는 주 내용을 처리하는 곳
'''
from flask import render_template, request, redirect, url_for, Response, jsonify
from service.controllers import bp_chat as chat
# from service.forms import FormQuestion
# 환경변수의 시크릿 키 획득을 위해서 Flask 객체 획득
from flask import current_app

from flask import Flask, render_template, jsonify, request
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel
import numpy as np
# from flask_socketio import SocketIO, emit
import random
from transformers import pipeline
from matplotlib.pyplot import imshow
import time
from transformers import TextClassificationPipeline
from transformers import BertTokenizerFast
from transformers import TFBertForSequenceClassification
from flask import current_app
from service.model.models import Information, Chatinformation
from service import db

# app = Flask(__name__)
# # 실시간 통신(SocketIO 사용)을 위해 비밀키 지정
# app.config['SECRET_KEY'] = 'dev'  # 임의의 키값을 지정
# socketio = SocketIO(app)

# 챗봇 사전학습된 모델이 저장된 경로
pretrained_path = './service/models/gpt_chatbot/'
# 토크나이저, 모델 로드
tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
model = TFGPT2LMHeadModel.from_pretrained(pretrained_path)
#-------------------------------------------------------------------------------------

# 감정 분류 모델
# kobert_감정분류 모델 경로
MODEL_SAVE_PATH = './service/models/kobert_emotion/'
# Load Fine-tuning model
loaded_tokenizer = BertTokenizerFast.from_pretrained(MODEL_SAVE_PATH)
loaded_model = TFBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)

text_classifier = TextClassificationPipeline(
    tokenizer=loaded_tokenizer,
    model=loaded_model,
    framework='tf',
    return_all_scores=True
)

#-------------------------------------------------------------------------------------

# 말투 변환 모델
model_name = "gogamza/kobart-base-v2"
nlg_pipeline = pipeline('text2text-generation',
                        model="./service/models/kobart_trans/",
                        tokenizer=model_name)

#-------------------------------------------------------------------------------------

# 챗봇 모델에 질문을 보내면 응답을 하는 함수
# 챗봇 응답


def request_answer_by_chatbot(text):
    # 1. 토크나이저를 이용하여 인코딩
    # </s><usr>질문<sys>
    bos = [tokenizer.bos_token_id]
    sentence = tokenizer.encode(
        f'{tokenizer.decode(2)}{text}{tokenizer.decode(4)}')
    sentence = bos + sentence
    # 2. sentence를 텐서로 변경
    # 학습의 n개의 문장으로 학습했으므로 형식을 2D로 맞춘다
    sent_tensor = tf.convert_to_tensor([sentence])
    # 3. 모델에 데이터 주입 -> 예측(=답변을 생성) 수행, 답변 50토큰까지 진행 => 패딩사이즈에 제한걸림
    #    텐서로 응답
    answer = model.generate(
        sent_tensor,
        do_sample=True,  # 샘플링 전략 사용
        max_length=100,  # 최대 디코딩 길이는 50
        top_k=10,  # 확률 순위가 10위 밖인 토큰은 샘플링에서 제외 (속도와 성능면에서 10으로 설정함)
        top_p=0.95,  # 누적 확률이 95%인 후보집합에서만 생성
        num_return_sequences=3  # 3개의 결과를 디코딩해낸다
    )
    # 4. 실데이터만 추출해서, 리스트로 변환 -> 디코딩 수행
    answer_list = []
    for a in answer:
        answer = tokenizer.decode(a.numpy().tolist())
        # 5. 실제응답한 내용만 추출
        # #print(answer)
        answer_list.append(answer.split(tokenizer.decode(4))
                           [-1].split(tokenizer.decode(1))[0].strip())
    return answer_list

# 감정분류


def predict(predict_sentence):

    text = predict_sentence
    # predict
    preds_list = text_classifier(text)[0]

    sorted_preds_list = sorted(
        preds_list, key=lambda x: x['score'], reverse=True)
    emotion_num = sorted_preds_list[0]['label']  # label
    # #print(emotion_num)
    predicted_score = sorted_preds_list[1]['score']  # score
    # #print(predicted_score)

    emotion_num = int(emotion_num.replace("LABEL_", "").strip())

    if emotion_num == 0:
        emotion_result = '불안'
    elif emotion_num == 1:
        emotion_result = '당황'
    elif emotion_num == 2:
        emotion_result = '분노'
    elif emotion_num == 3:
        emotion_result = '슬픔'
    elif emotion_num == 4:
        emotion_result = '기쁨'
    elif emotion_num == 5:
        emotion_result = '상처'
    # label, 확률값 리턴
    return emotion_result, predicted_score

# 말투 변환


def generate_text(pipe, text, target_style, num_return_sequences=5, max_length=60):
  # 말투 스타일 리스트
  style_list = {'choding' : '초등학생', 'joongding' : '중학생', 'halbae' : '할아버지', 'halmae' : '할머니'}
  target_style_name = style_list[target_style]
  text = f"{target_style_name} 말투로 변환:{text}"
 # #print(text)
  out = pipe(text, num_return_sequences=num_return_sequences,
             max_length=max_length)
  return [x['generated_text'] for x in out]


def main(question, caccent):
    # 설정
    # 캐릭터 선택
    character = caccent
    ##print('\n')
    ##print("   설정이 완료되었습니다. 채팅을 시작합니다.")
    ##print('\n')

    # 감정 지수 리스트
    emotion_list = {
        "불안": 0,
        "당황": 0,
        "분노": 0,
        "슬픔": 0,
        "기쁨": 0,
        "상처": 0,
    }

    # 챗봇
    sentence = question
    user_emotion, user_p = predict(sentence)
    #print(f'    label:{user_emotion}, 예측 확률:{user_p * 100:.2f}%')
    emotion_list[user_emotion] += 1
    #print("\n")
    answer_3 = request_answer_by_chatbot(sentence)

    max_p = 0
    max_s = " 가득하네요."  # 임의 답변
    max_l = user_emotion

    # 기쁨은 기쁨으로만 답변
    if user_emotion == "기쁨":
        for a in answer_3:
            l, p = predict(a)
            # #print(a)
            if l == "기쁨":
                if p > max_p:
                    max_p = p
                    max_s = a
                    max_l = l
    else:
        for a in answer_3:
            l, p = predict(a)
            # #print(a)
            if l != "기쁨":
                if p > max_p:
                    max_p = p
                    max_s = a
                    max_l = l

    if max_s == " 가득하네요.":
        max_s = user_emotion + max_s

    max_s = generate_text(nlg_pipeline, max_s, character,
                            num_return_sequences=1, max_length=50)[0]
    #plt_imshow('Jieun', img=[image_dic[max_l]], figsize=(8, 5))
    ##print(f'    답변 : {max_s}')
    ##print(f'    label:{max_l}, 예측 확률:{max_p * 100:.2f}%')
    ##print("\n")
    #time.sleep(7)
    result = {}
    result['q_e'] = user_emotion
    result['a'] = max_s
    result['a_e'] = max_l
    return result

# 모델 로드 완료
# ------------------------------------------------------------------------------------------------------------------------------------

# 플라스크 객체뿐 아니라 등록한 blue#print 객체도 라우팅 가능
# ~/chat/


@chat.route('/', methods={'POST', 'GET'})
def chat_home():
    if request.method == 'GET':
        # a = id_search()
        # if not a:
        #     return render_template('login.html')
        return render_template('chat.html')
    else:
        check = request.form.get('msg')
        if check == "end":
            return jsonify({'code': 1})
        else:
            return jsonify({'msg': "채팅이 종료되지 않았습니다. 다시 시도해주세요."})

# 1.챗봇 질문에 대한 답변 처리
# 2. 챗봇 질문에 대한 감정 분류
# 3. 말투 변환
# 보류 - 전송된 질문을 질문큐로 전송 -> 에이전트가 예측( 답변 생성 ) -> 응답큐로 전송 (시간 체크해야 함 time 함수)
# 디비 사용 검토


@chat.route('/chat_proc', methods=['POST'])
def chat_proc():
    # 추후 딥러닝 모델을 추가해서 응답을 동적으로 예측(문장생성)하여 처리한다
    question = request.form.get('msg')  # 사용자의 질문을 추출
    # 토큰 아이디 불러오기
    id = id_search()
    info2 = Information.query.filter(Information.id==id).all()
    result = main(question, info2[0].caccent)  # 질문->모델예측->답변생성
    # print(info2)
    # s3에 저장된 이미지는 영어 감정이므로 변경
    emotion_trans_list = {"분노":"Angry", "불안":"Fear", "기쁨":"Happy", "슬픔":"Sad", "상처":"Sad", "당황":"Surprise"}
    en_emotion = emotion_trans_list[result['a_e']]
    url = f'https://d2aceluorl5wri.cloudfront.net/GANS/datas_results/{str(id)}/{en_emotion}.jpg'
    res = {'url' : url, 'answer': result['a'], 'name': info2[0].cname}  # 답변을 넣어서 응답
    # 아이디 중복확인
    # chatinfo = Chatnformation.query.filter(Chatnformation.uid==uid).all()
    #bupw = (bcrypt.hashpw(upw.encode('utf-8'), bcrypt.gensalt())).decode('utf-8')
    # 감정을 어떻게 넣을 것인가 찾기
    a = Chatinformation(
        id=id, question=question, answer=result['a'], que_emotion=result['q_e'], ans_emotion=result['a_e'], chat_date=datetime.now())
    # 신규 가입 등록
    db.session.add(a)
    # 커밋
    db.session.commit()
    return jsonify(res)

# # 클라이언트 메시지를 처리하기 위한 이벤트 등록및 처리 - 채팅 답변
# @socketio.on('cTos_simple_msg')
# def cTos_simple_msg(data):
#     #print(data)
#     # echo, 받는 내용을 살짝 수정해서 응답(서버 => 클라이언트로 메시지 전송, 푸시)
#     data['msg'] += "<응답"
#     emit('sToc_simple_msg', data)

# 채팅종료
@chat.route('/chat_end')
def chat_end():
    return render_template('chat_end.html')

@chat.route('/chat_end_predict', methods={'POST'})
def chat_end_predict():
    # if request.method == 'GET':
    #     return render_template('chat_end.html')
    # else:
    # 감정 분석 결과 사용 문구
    text_list = {
        "불안": "지금 네가 걸어가고 있는 길에서 고민되고<br>불안한 생각이 들 때도 있겠지만<br>넌 지금도 충분히 잘하고 있어.<br>그러니 더 잘하겠다는 생각을 하지 않아도 괜찮아.<br>오늘 하루도 애썼어, 너의 내일을 응원할게",
        "당황": "하루가 저무는 것처럼 걱정도 저무는 밤길.<br>모든 근심 걱정 내려놓고 편안한 밤 되세요.",
        "분노": "혼자 있을 땐 심호흡을 하면서 자신을 진정시켜야 해요.<br>'별 것 아냐' '괜찮아' 등과 같은 혼잣말을 하는 것도 방법이에요.<br>클래식과 같은 편안한 음악을 듣거나 일기를 쓰는 것도 마음의 안정을 찾는데 도움을 줄 것이에요.",
        "슬픔": "행복이란 내가 갖지 못한 것을 바라는 것이 아니라<br>내가 가진 것을 즐기는 것이다.",
        "기쁨": "오늘 하루는 어떠셨나요? 즐거운 일들이 많이 있으셨나요?<br>좋은 기억은 간직하시고 좋지 않은 기억은<br>저무는 해에 날려 보내시길 바랍니다.",
        "상처": "가장 배우기 어려운 교훈은<br>우리에게 상처를 안겨준 자들을 용서하는 것이다.<br>- 조셉 자콥스",
    }
    # 토큰 아이디 불러오기
    id = id_search()
    info = Chatinformation.query.filter(Chatinformation.id==id).all()
    total_emotion = {"불안": 0, "당황": 0, "분노": 0, "슬픔": 0, "기쁨": 0, "상처": 0}
    for row in info[::-1]:
        # 24시간 채팅 내용만 분석
        if row.chat_date < (datetime.utcnow() - timedelta(seconds=60*60*24)):
            break
        total_emotion[row.que_emotion] += 1
    max_key = max(total_emotion, key=total_emotion.get)
    info2 = Information.query.filter(Information.id == id).all()
    if total_emotion[max_key] > 0:
        msg = f"오늘 {info2[0].uid}님은 {max_key} 가득하네요.<br><br>{text_list[max_key]}"
    else:
        msg = "결과 분석에 실패하였습니다. 다시 채팅을 시도해주세요."

    res = {'msg': msg}
    return jsonify(res)

def id_search():
    # 접속 아이디 정보 불러오기
    token = request.cookies.get('token')
    SECRET_KEY = current_app.config['SECRET_KEY'] 
    payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    id = payload['id']
    return id


