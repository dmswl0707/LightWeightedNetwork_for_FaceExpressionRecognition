# LightWeightedNetwork_for_FaceExpressionRecognition
 ![image](https://user-images.githubusercontent.com/65028694/130179867-3b5f14ff-12bc-4d38-b8b6-dc14b075b0c0.png)
 
자동 얼굴 표정인식(FER) 분야는 인간과 컴퓨터의 상호작용(HCI), 스마트 홈 IoT, UAV 및 로봇틱스(Robotics), 고급 운전자 지원 시스템 영역에 다양한 응용 프로그램을 제공합니다. 그러나, 기존 네트워크에서 요구하는 높은 연산량과 느린 연산 속도로 인해, 그 적용범위가 제한되고 있다. 우리는 기존 인공신경망에서 요구되었던 고성능 GPU환경과 높은 연산량을 극복하고자 모델 경량화(Light weighted Model) 기법을 적용하여 IoT 및 모바일 기기에서 적용될 수 있는 얼굴 표정 인식 신경망을 제안합니다. 



### Network Architecture
![image](https://user-images.githubusercontent.com/65028694/130179990-560f564d-9e47-4bed-81d6-1ac827e5a9d3.png)



### Usage
> preprocessing.py 데이터 전처리
>
> dataloader.py 데이터 로드
> 
> face detection.py 웹캠으로 얼굴 인식 후 정면을 48*48사이즈로 크롭하여 이미지 저장.
> 
> model.py 커스텀 CNN 신경망
> 
> train.py 신경망 학습
> 
> #### Inference 
> test_class.py 클래스별 정확도 검출
> 
> test_ConfusionMatrix.py 혼동행렬 결과 출력

