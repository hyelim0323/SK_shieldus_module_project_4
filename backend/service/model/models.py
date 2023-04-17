from service import db

class Information(db.Model):
    id      = db.Column(db.Integer, primary_key=True)   # 고객 관리 ID
    uid   = db.Column(db.String(32), nullable=False, unique=True)    # 로그인 ID
    upw   = db.Column(db.CHAR(60), nullable=False)    # 로그인 PW
    cname   = db.Column(db.String(32), nullable=False)    # 캐릭터 닉네임
    caccent = db.Column(db.String(128), nullable=False)    # 캐릭터 말투
    reg_date = db.Column(db.DateTime(), nullable=False) # 고객 가입일
    up_date = db.Column(db.DateTime(), nullable=False) # 회원 정보 수정일


class Chatinformation(db.Model):
    index = db.Column(db.Integer, primary_key=True) 
    id    = db.Column(db.Integer, nullable=False)   # 고객 관리 ID
    question   = db.Column(db.Text(), nullable=False)    # 질문
    answer   = db.Column(db.Text(), nullable=False)    # 답변
    que_emotion   = db.Column(db.String(32), nullable=False)    # 질문 감정
    ans_emotion = db.Column(db.String(32), nullable=False)    # 답변 감정
    chat_date = db.Column(db.DateTime(), nullable=False) # 채팅 일자
    # url 추개
    pass

