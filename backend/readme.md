# 컨테이너 접속해서 다음 명령어 실행
    docker exec -it module_project_4-backend-1 bash 
    flask --app service db init
    flask --app service db migrate
    flask --app service db upgrade