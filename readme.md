# 웹 페이지
![로그인_회원가입](https://user-images.githubusercontent.com/72790897/233084521-42838112-08a4-40d2-8e10-23499f52f659.png)
![채팅](https://user-images.githubusercontent.com/72790897/233084530-96f3c502-e12e-42a5-93a5-3a14182e9fa0.png)

# 시스템 구조
## 1.
![참고_시스템구성도 drawio의 사본 (1)](https://user-images.githubusercontent.com/72790897/233085261-fabae363-1cec-4d9a-94bf-dddd0c8731fc.png)
*회원가입
- 아이디, 비밀번호, 챗봇이름, 챗봇말투 기입 -> db
- 이미지 업로드 -> s3
- s3 trigger 발동 -> agent가 업로드된 이미지를 변환하여(GAN 적용) 결과를 s3에 업로드

*로그인
- db에서 확인 -> 로그인되면 토큰 발급

## 2.
![참고_시스템구성도 drawio의 사본 drawio](https://user-images.githubusercontent.com/72790897/233085271-ac875a04-adb0-4847-afba-5f6959227a4d.png)
*채팅
- 질문 -> was에서 답변생성 및 응답 (db 삽입)

*채팅 결과
- was에서 token의 id과 일치하는 채팅 기록을 db에서 불러옴
- 24시간 내에 기록된 내역을 기반으로 감정 결과 분석


# 트리구조
  - Tier-3
    - agent     # 이미지 변환
    - proxy     # web 포지션, nginx
    - backend   # was 포지션, flask
    - db        # db 포지션, db
    - docker-compose.yml    # 컴포즈 정의 파일
    - readme.md     # 설명파일
  
## 도커 컴포즈 명령어
docker-compose up -d
### DB 설치
docker exec -it module_project_4-backend-1 bash
flask --app service db init
flask --app service db migrate
flask --app service db db
