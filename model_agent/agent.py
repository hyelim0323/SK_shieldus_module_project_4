import boto3
import os
import tensorflow as tf
from tensorflow.keras import models
import urllib
from PIL import Image
import numpy as np
import time 
import json
from cloudpathlib import CloudPath
# sudo yum install -y mesa-libGL libgl1 libglib2.0-0
# pip3 install opencv-python-headless==4.5.1.48
import cv2
import torch
import torchvision.transforms as transforms

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
sqs = boto3.client('sqs')

# 폴더 생성
def createDirectory(directory):
    print(directory)
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
        
# 초기 모델 파일 다운로드
def init_weight():
    # 폴더 다운로드
    f_list = ['StarGAN/stargan_custom/models', 'CartoonGAN/사전학습모델/pretrained_model']
    if not os.path.exists('./agent/' + f_list[0]):    # 파일이 없다면
        cp1 = CloudPath("s3://ai-project4-group6/GANS/StarGAN/")
        createDirectory('./agent/' + f_list[0])
        cp1.download_to('./agent/' + f_list[0])
        print('다운로드 완료')
    # elif not os.path.exists('./agent/' + f_list[1]):    # 파일이 없다면
    #     cp1 = CloudPath("s3://ai-project4-group6/GANS/CartoonGAN/")
    #     createDirectory('./agent/' + f_list[1])
    #     cp1.download_to('./agent/' + f_list[1])
    #     print('다운로드 완료')
    else:
        print('이미 존재함')
  
# 이미지 전처리 (얼굴 인식)
def face_detection(url, img_path):
    # 이미지 다운로드
    print("333")
    # print(url)
    # print(img_path)
    p = img_path.split('/')
    image_path = "./agent/StarGAN/data" + "/" + p[-2]
    createDirectory(image_path)
    print(image_path)
    urllib.request.urlretrieve(url, image_path + '/' + p[-1])
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('agent/StarGAN/stargan_custom/models/haarcascade_frontalface_alt2.xml')
    
    # 이미지 변환
    img = cv2.imread(image_path + '/' + p[-1], cv2.IMREAD_COLOR)
    
    ## 얼굴 crop -> 256X256 resize
    # grayscale로 변환
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # print(image)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    
    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
      # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
      crop_img = img[y:y + h, x:x + w]
      crop_img = cv2.resize(crop_img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
      break
  
    gan(crop_img, p[-2])

# # CartoonGAN 적용
# # make_cartoon
# def make_cartoon(img, style="Shinkai"):
#     # print(img)
#     from CartoonGAN.사전학습모델.network.Transformer import Transformer
#     model = Transformer()
#     model.load_state_dict(torch.load(os.path.join('agent/CartoonGAN/사전학습모델/pretrained_model', style + '_net_G_float.pth')))
#     model.eval()
#     # print('{} Model loaded!'.format(style))
#     img_size = 256
#     T = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize(img_size, 2),
#     transforms.ToTensor()])
 
#     img_input = T(img).unsqueeze(0)
#     img_input = -1 + 2 * img_input
    
#     img_output = model(img_input)
 
#     img_output = (img_output.squeeze().detach().numpy() + 1.) / 2.
#     img_output = img_output.transpose([1, 2, 0])
#     print("cartoonGAN까지 생성완료!")
#     return img_output
    
    
# StarGAN 결과 이미지 생성 및 CartoonGAN으로 변환 이미지 생성 후 결과 s3로 송신
def gan(img, user_id) : 
    from StarGAN import main as s_main
    s_main.main_real(user_id)
    
    stargan_img_dir_path = "agent/StarGAN/stargan_custom/results"
    stargan_img_dir_list = os.listdir(stargan_img_dir_path)
    print(stargan_img_dir_list)
    
    img_dict = {}
    
    for i_file_name in stargan_img_dir_list:
        image = cv2.imread(stargan_img_dir_path + "/" + i_file_name, cv2.IMREAD_COLOR)
        # cartoon 이미지로 변환
        # Shinkai 스타일 
        c_img = image # make_cartoon(image, style="Shinkai")
        # cv2.imwrite(stargan_img_dir_path + "_/" + i_file_name, c_img)
        print(stargan_img_dir_path + "_/" + i_file_name)
        # 이미지 s3 송신 (업로드)
        bk = 'ai-project4-group6'
        # 업로드시 충돌나지 않게 이름을 해싱한다, 여기서는 그냥 이름
        with open(stargan_img_dir_path + "/" + i_file_name, 'rb') as f:
            data = f.read()
        s3.Bucket(bk).put_object(Key='GANS/datas_results/' + user_id + '/' + i_file_name, Body=data)
    
# 실행함수          
def execute():
    # 대기열 큐를 특정 주기로 계속해서 체크, 모니터링, 감지한다
    q_name = 'project4_sqs_1'
    res = sqs.receive_message(
        QueueUrl=q_name,
        AttributeNames=[
            'SentTimestamp'
        ],
        MessageAttributeNames=[
            'All',
        ],
        MaxNumberOfMessages=1,  # 1개씩 가져오는것으로
        VisibilityTimeout=0,
        WaitTimeSeconds=0,
        )
    if res and ("Messages" in res):   # 해당 키가 존재하면 메세지가 존재하는 것
        # 수신한 값을 기준 => 메세지가 존재하는 여부 체크 -> 메세지 삭제(큐에서) -> 예측 처리 요청 진행
        receipt_handle = res["Messages"][0]['ReceiptHandle']   # 메세지 고유값
        body = res["Messages"][0]["Body"] 
        body = json.loads(body)
        print( body['data'] )
        standby_predict( body['data'] )
        
        # 메세지 삭제 -> 큐에서 제거
        sqs.delete_message(
            QueueUrl=q_name,
            ReceiptHandle=receipt_handle
            )
        print('메세지 풀링')
        # 예측 수행을 지시(컨테이너) -> 예측 수행(딥러닝 에이전트 or 람다함수) 
        # -> 결과를 받아서 응답 하는 큐에 메세지를 전송
    else:
        print('no message')
 
# 예측 준비        
def standby_predict( key ):
    # sqs => 메세지 획득 => key 획득 => 예측수행
    cdn = 'https://d2aceluorl5wri.cloudfront.net'
    #url = f'{cdn}/data/{key}'
    url = f'{cdn}/{key}'
    print("key", key)
    face_detection( url, key )
    
if __name__=="__main__":
    init_weight()
    while True:
        try:
            execute()
        except Exception as e:
            print(e)
        time.sleep(1)