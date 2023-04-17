FROM python:3.9.16-bullseye AS builder


# 작업 디렉토리 지정, 없으면 생성한다
WORKDIR /was_app

# 1개의 컨테이너에는 1개의 소프트웨어를 담는다
# 가상환경이 필요없다
# 필요한 설치 패키지를 기술한 파일을 ADD or COPY
COPY requirements.txt /was_app

RUN pip3 install -r requirements.txt

COPY . /was_app

# 환경변수, 앱의 이름은 flask run 명령시 자동 인식하는 이름이므로 생략
# 기본 이름이 아닌 경우 엔트리포인트를 직접 지정
ENV FLASK_APP service
# 차후 프로덕션인 경우 수정
ENV FLASK_ENV development
ENV FLASK_RUN_PORT 8000
ENV FLASK_RUN_HOST 0.0.0.0

# 포트 설정
EXPOSE 8000

# 구동 명령
ENTRYPOINT [ "flask" ]

# 옵션
CMD ["run"]

