# 웹 페이지
https://user-images.githubusercontent.com/72790897/233084521-42838112-08a4-40d2-8e10-23499f52f659.png
https://user-images.githubusercontent.com/72790897/233084530-96f3c502-e12e-42a5-93a5-3a14182e9fa0.png

# 시스템 구조
![인프라구성도1](https://user-images.githubusercontent.com/72790897/233083614-b9d15fcf-2429-40ea-b9e3-3af6418ff105.JPG)
![인프라구성도2](https://user-images.githubusercontent.com/72790897/233083623-714f7c23-e98d-4cc5-8501-6afd2f837b84.JPG)

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
