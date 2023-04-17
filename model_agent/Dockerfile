# docker build --no-cache -t agent:1.0 ./model_agent
# botocore.exceptions.NoCredentialsError: Unable to locate credentials 해결
# docker run -itd --name agent_model -v ~/.aws/:/root/.aws:ro --env AWS_PROFILE=default agent:1.0 
# docker logs agent_model
# docker container stop agent_model
# docker container rm agent_model
# docker image ls
# docker image rm agent:1.0
FROM python:3.9.16-bullseye

WORKDIR /model_agent

COPY requirements.txt /model_agent

RUN pip3 install -r requirements.txt
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get install libglib2.0-0

COPY agent.py /model_agent

ENTRYPOINT [ "python" ]
CMD ["agent.py"]