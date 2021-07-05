# LightWeightedNetwork_for_FaceExpressionRecognition

### dataset
CK+ 데이터셋을 사용, 589개의 test 이미지, 196개의 validation 이미지, 196개의 test 이미지로 구성

### model
![image](https://user-images.githubusercontent.com/65028694/124432560-b8d4cf80-ddac-11eb-94fb-771c70775237.png)

### 실험결과
본 실험은 Lr 0.001, SGD optimizer, Epoch 200 환경에서 진행하였으며, Loss는 Cross Entropy Loss를 사용하였다. 실험 결과는 Batch size를 8로 설정하고, ELU를 사용할 때, 97.83(%)로 본 네트워크 실험에서 결과가 가장 좋았다. Vanilla Convolution Network와 비교할 때, 정확도는 1.47% 부족한 결과를 보이나, 1/230배 가량 적은 파라미터를 가진다.

![image](https://user-images.githubusercontent.com/65028694/124432697-dd30ac00-ddac-11eb-872e-37b92cd901bc.png)

### inference 
test_class.py 사전 학습된 가중치를 불러와 class 별로 정확도를 출력

test_ConfusionMatrix.py 각 클래스 별로 true label과 predict label 차이를 Confusion Matrix로 출력


![image](https://user-images.githubusercontent.com/65028694/124432256-65628180-ddac-11eb-989d-8792735f0bdc.png)
(ck+ testset 클래스 별 구성)

Anger 이미지 27장, Contempt 이미지 10장, Disgust 이미지 35장, Fear 이미지 15장, Happy 이미지 41장, Sadness 이미지 18장, Surprise 이미지 50장으로, 총 196장

face_detection.py 웹캠으로 얼굴 인식 후 정면을 48X48 사이즈로 크롭하여 폴더에 저장.(웹캡으로 inference시)

train.py 데이터셋을 모델에 학습하여 .pt 파일에 가중치를 저장.
