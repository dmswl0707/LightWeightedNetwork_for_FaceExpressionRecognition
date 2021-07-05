# LightWeightedNetwork_for_FaceExpressionRecognition

### dataset
CK+ 데이터셋을 사용, 589개의 test 이미지, 196개의 validation 이미지, 196개의 test 이미지로 구성

### inference 
test_class.py 사전 학습된 가중치를 불러와 class 별로 정확도를 출력
test_ConfusionMatrix 각 클래스 별로 true label과 predict label 차이를 Confusion Matrix로 출력
(testset 클래스 별 구성)
Anger 이미지 27장, Contempt 이미지 10장, Disgust 이미지 35장, Fear 이미지 15장, Happy 이미지 41장, Sadness 이미지 18장, Surprise 이미지 50장으로, 총 196장
![image](https://user-images.githubusercontent.com/65028694/124432256-65628180-ddac-11eb-989d-8792735f0bdc.png)

### model
![image](https://user-images.githubusercontent.com/65028694/124432560-b8d4cf80-ddac-11eb-94fb-771c70775237.png)

