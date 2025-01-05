### 1강 : Intro
- 활성함수 : DL의 꽃
- 정규화 : 오버피팅을 피하는 방법
- CNN : 이미지 데이터 처리
- RNN : 시계열 데이터 처리
- Attention, Transformer : 시계열 데이터, 자연어 처리

### [실습] PyTorch 환경 설정하기 (Mac)
- python3 설치, 환경변수 설정
  - python 3.12.8 다운로드, 설치
  - 기존 3.10 환경변수 폴더내 pip ~ python* 파일 제거
  - 3.12 환경변수 폴더 들어가서 python.exe -> python3.exe 복사
  - $ python3 --version 확인
  - Python 3.12.8
  - miniconda3(=anaconda) 24.11.0

- miniconda3 설치, 환경변수 설정
  - miniconda3 24.11.0 다운로드, 설치
  - C:\ProgramData\miniconda3 -> _conda.exe -> conda.exe 복사
  - 시스템 변수 > Path > C:\ProgramData\miniconda3 추가
  - $ conda --version 확인
  - conda 24.11.0

- conda env 환경 만들기
  - $ conda create -n py312 python=3.12 --yes
  - [Python] conda 가상환경 activate 오류
  - git bash 실행
  - $ source /c/ProgramData/miniconda3/etc/profile.d/conda.sh
  - $ conda activate py312
  - $ conda install pytorch torchvision torchaudio -c pytorch --yes
  - $ pip install tqdm jupyter jupyterlab scikit-learn scikit-image tensorboard torchmetrics matplotlib pandas

### [실습] PyTorch 환경 설정하기 (Windows)
- Miniconda 설치
  - google : install miniconda windows 검색
  - Windows Miniconda3 Windows 64-bit 다운로드, 설치
    - 인스톨 > 모든 거 다 체크하고 설치
  - 윈도우 검색 > Anaconda Prompt (miniconda3) 실행

- Programming IDE 설치
  - VS Code 설치

- PyTorch와 기타 library 설치
  - Anaconda Prompt 실행
  - py312 가상환경 만들기
    - $ _conda create --name py312_windows python=3.12 --yes
    - $ _conda activate py312_windows
    - $ _conda install pytorch torchvision torchaudio -c pytorch --yes
    - $ python
      - >>> import torch
      - >>> exit()
    - $ pip install tqdm jupyter jupyterlab scikit-learn scikit-image seaborn tensorboard torchmetrics torchinfo matplotlib pandas

### [실습] Colab으로 PyTorch 환경 설정하기
- Google Drive 접속
- Google Colaboratory 실행
- Notebook 시작하기

[1]
from google.colab import drive
drive.mount("/content/drive")
Mounted at /content/drive

[2]
!pwd
/content

[3]
!ls
drive  sample_data

[5]
import os
os.chdir("/content/drive/MyDrive")

[6]
!ls
'apache ni-fi for dummies 번역본 모르는 것.gdoc'
'BAS Cluster'
'BAS Stand alone'
'BAS 설치 순서 history.gdoc'
 bis
'Colab Notebooks'
 DIET_BACKUP.zip
'Hadoop 완벽 가이드 4판 - 모르는 것 체크.gdoc'
 olmizuki_20240413.mp4
 RDS
'windows10 hosts 설정법.gdoc'
'xxf-pqir-des - 2021년 5월 24일.gjam'
'빅데이터 공유, 발표 자료 기획 to 한미마그 신입.gdoc'
'빅데이터교육자료(2021 3 2)'
'소득세액공제신고서(개정안).hwp'
 시와소프트
 이종화
'일일업무(이종화)_202012.xlsx'
 자기소개서_코딩온_웹개발자_입문_부트캠프_7기_한지인.docx
'제목 없는 문서 (1).gdoc'
'제목 없는 문서.gdoc'
'제목 없는 스프레드시트 (1).gsheet'
'제목 없는 스프레드시트 (2).gsheet'
'제목 없는 스프레드시트 (3).gsheet'
'제목 없는 스프레드시트.gsheet'
'제목 없는 프레젠테이션.gslides'
'현대오토에버_3분기_ICT대규모채용_이종화(트라닉스 웹 운영) (1).gslides'
'현대오토에버_3분기_ICT대규모채용_이종화(트라닉스 웹 운영).gslides'

[7]
import torch

[8]
!pip install scikit-learn
Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (1.6.0)
Requirement already satisfied: numpy>=1.19.5 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.26.4)
Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.13.1)
Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.4.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (3.5.0)

### [이론] 딥러닝 (Deep Learning)은 뭘까?
- 딥러닝이란?
  - Deep Learning이란 무엇? 어디에서 기원하는가?
    - 딥러닝으로 어떤 문제를 풀 수 있을까?
      - Input Data X -> Model -> Y = Cat
      - 입력값 X -> Label Y의 값은 Y = f(X)
      - X -> Y로 매핑해주는 함수 f(X)는 어떻게 구할까?
        - 정확한 f(X) 구하기는 어렵다.
      - f(X)에 "근사"하는 f^(X)으로 모델링 한다. --> Y^ = NN(X), f(X) ~ f^(X)
        - 목표 : Y^=NN(X) 뉴럴 네트워크의 출력값이 Y=f(X) 실제값에 최대한 유사하도록 하는 것. (뉴럴넷을 학습한다는 의미임.)

  - 어떤 문제를 풀려고 하는 것인가?

  - Neural Network의 기본 구성은 어떻고, 어떻게 학습시키는가?

### [이론] 뉴럴넷 (Neural Network)은 뭘까?
- 기본적인 Neural Network의 구성
  - 기본 구성 : Input Layer + (한 개 이상의) Hidden Layer + Output Layer
  - 특성 :
    - 각 Layer은 뉴런(Neuron) 들로 구성됨.
    - 각 뉴런은 다음 Layer의 뉴런과 연결 (edge) 되어 있음
    - 즉, 뉴런은 이전 Layer의 뉴런들의 출력값으로 입력을 받는다.
    - 각 뉴런은 가중치 weight w와 활성 함수 activation fucntion으로 구성된다.

  - 예시 :
    - x1, x2, x3
      - 뉴럴넷에 입력되는 값
      - input layer의 1, 2, 3번째 뉴런이 출력하는 값
      - 다음 Layer의 Input 값으로 전달
    - hj
      - Hidden Layer의 j번째 뉴런이 출력하는 값
      - 이전 Layer인 Input Layer의 출력값을 입력값으로 사용하여 계산됨
      - 다음 Layer의 Input 값으로 전달

  - x1, x2, x3 => hj 계산과정
    - 입력 값 : x1 / x2 / x3 (input)
    - 출력 값 : hj (output)
    - 과정 :
      - Weight Multiplication (가중치 곱) -> Aggregation (집계) -> Activation Function (활성 함수)
    - 상세 :
      - Weight Multiplication : x1 * wj1 / x2 * wj2 / x3 * wj3
      - Aggregation : (x1 * wj1) + (x2 * wj2) + (x3 * wj3)
      - Activation Function : AF(Aggregation결과) => hj 출력

- Neural Network의 기원
  - 뇌의 뉴런에서 시작
    - Input : Dendrite(수상돌기) - 다른 뉴런의 출력 Singal 신호를 수용
    - 중간 과정 : 뉴런의 특성(~ activation function과 비슷)
    - Output : Axon Terminal(뉴런말단) - 출력 Signal

### [실습] Neural Network의 동작 원리
- Question : 어떻게 해야 뉴럴 네트워크의 출력값 Y^가 실제 값인 Y에 근사 할까?
  - 각 Layer의 weight w를 최적회해야함! (= 각 Layer에 곱해지는 가중치의 최적 조합을 찾고 싶은 것!)
    - Weight wji이 바뀌면 주어진 input xi값에 대한 Output y 값도 바뀐다.
    - 주어진 Input xi들에 대해서 최대한 실제값 y과 유사하게 Output y^을 출력해주는 가중치 wji들의
      조합을 찾고 싶은 것
    - 각 Layer의 weight wji을 적절하게 조정해서 주어진 Input에 대해서 출력되는 y^이 실제값 y에 최대한
      잘 근사하도록 최적화 하는 것

  - Weight 값을 어떻게 정의해야 예측값이 최대한 정확할까?
    - 기본적으로 알 수 없음, 처음에는 랜덤하게 정의해야함

  - Weight 값을 어떻게 최적화해야 모델의 예측값이 더 정확해질 수 있을까?
    - Gradient Descent (경사 하강)을 통한 Loss fucntion (손실 함수)값을 최소화하도록, weight값을
      최적화하여 점진적으로 모델의 예측 정확도를 높인다.

- Answer : 즉, 처음에는 랜덤한 weight 값에 따라 모델의 예측값도 random 하지만, weight w 값을
  최적화하여 점차 모델의 정확도를 높인다!

### [실습] Deep Learning 실무 기초 개념
-   **Training, Validation, Test dataset**
    -   **Training (학습/훈련) 데이터셋**
        -   모델을 학습시키는 용도
        -   딥러닝 모델의 경우 학습 데이터셋에 대해서 Gradient Descent하여 Loss가 최소화되도록 모델의 weight를 최적화함
    -   **Validation (검증) 데이터셋**
        -   모델의 성능 평가와 Hyperparameter Tuning에 사용
            -   **Hyperparameter?**
                -   모델 구조와 학습 방식에 영향을 주는 파라미터, 모델의 최종 성능에도 영향을 줌
            -   **Hyperparameter Tuning?**
                -   가장 최적의 Hyperparameter 조합을 찾는 것.
                -   Validation 성능이 가장 높은 조합 찾기
        -   학습 데이터셋에 대한 "Overfitting" 될 수 있기에 Validation 데이터셋으로 성능 평가
            -   **Overfitting(과적합)이란?**
                -   Unseen data에 대해서 모델의 예측값이 일반화되지 않는 경우
                -   뉴럴넷 모델이 학습 데이터에 있는 noise에 대해서도 학습하여, 일반화 성능이 저하되는 현상
                -   Epochs(반복 학습)이 늘어날 수록 성능이 마냥 좋이지지 않는다. 어느 기점부터 Valid Loss가 증가하는 현상이 발생하는데 이를 과적합이라고 함
                    -   예: 데이터 자체의 노이즈를 고려하지 않고 모델의 Degree를 마냥 높이면 실제 True f(x)와의 오차가 심하게 발생함
    -   **Test (시험) 데이터셋**
        -   검증 단계에서 선택한 최적의 모델의 최종 성능을 평가하는데 사용
        -   Hyperparameter Tuning을 과도하게 적용하는 경우, Validation dataset에 대해서 unintentional overfitting이 발생할 수 있다.

-   **Overfitting**

-   **K-fold Cross Validation**: 데이터셋이 너무 작아서 Validation dataset을 확보할 수 없을 때 사용
    1.  **데이터 분할**: 원본 데이터셋을 K 개의 서로 겹치지 않는 부분 집합으로 나눈다. (일반적으로 K=5 혹은 10 사용)
    2.  **반복 학습 및 검증**
        -   하나의 Fold를 검증 데이터로 사용
        -   K-1 개의 폴드를 훈련 데이터로 사용하여 모델 학습
        -   i) K = 6
        -   | Validation | Training | Training | Training | Training | Training |
            | --- | --- | --- | --- | --- | --- |
            | (Fold 1) | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 |

        -   ii)

            | Training | Validation | Training | Training | Training | Training |
            | --- | --- | --- | --- | --- | --- |
            | Fold 1 | (Fold 2) | Fold 3 | Fold 4 | Fold 5 | Fold 6 |

        -   iii)

            | Training | Training | Validation | Training | Training | Training |
            | --- | --- | --- | --- | --- | --- |
            | Fold 1 | Fold 2 | (Fold 3) | Fold 4 | Fold 5 | Fold 6 |

        -   iv)

            | Training | Training | Training | Training | Training | Validation |
            | --- | --- | --- | --- | --- | --- |
            | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | (Fold 6) |


3.  **성능 측정**
    -   각 반복마다 모델은 검증용 폴드에 대한 성능 평가
    -   모든 K 번의 평가 완료 후에 평균 성능 계산

4.  **최종 성능 평가 혹은 Hyperparameter Tuning**
    -   Fold들에 대한 평균 성능을 최종 성능 지표로 사용
    -   혹은 Hyperparameter Tuning에 사용할 수도 있음

5.  **장점**
    -   데이터를 효과적으로 활용하여 모델의 성능을 평가
    -   과적합 문제를 예방하고 모델의 일반화 성능을 더 신뢰성 있게 추정
    -   검증 데이터의 선택에 따라 모델의 평가 결과가 크게 변하지 않는다.

-   **Hyperparameter Tuning**

-   **Loss와 Evaluation Metric의 차이**

    -   **Loss Function**
        -   예측한 값과 실제 타깃 값 사이의 차이
        -   학습 단계에서 Loss를 최소화하는 방향으로 모델의 Weight를 조정
        -   미분 가능해야 함 (Gradient Descent를 사용하기 위해)
        -   Cross Entropy Loss, Mean Squared Loss 등
        -   **결론**: 모델 학습 단계에서 모델의 가중치를 조정하기 위한 목적으로 예측 오차를 측정

    -   **Evaluation Metric**
        -   학습된 모델의 성능을 평가하는데 사용되는 지표
        -   손실 함수와 달리 평가 지표는 더 직관적이다.
        -   정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1 Score 등
        -   **결론**: 학습된 모델의 성능을 평가하고 보고하기 위해 사용

### [실습] PyTorch 기초 - Tensor
-   **PyTorch의 Tensor**
    -   **Tensor (torch.Tensor)**
      -   Numpy의 배열 (array)와 행렬 (matrix)와 매우 유사한 자료구조
      -   PyTorch에서 scalar, vector, matrix, tensor등을 표현하는데 사용

      -   예:
          -   Scalar : 1
              -   1

          -   Vector : (2)
              -   [1]
                  [2]

          -   Matrix : (2, 3)
              -   [[1 3 5]]
                  [[2 4 6]]

          -   Tensor : (2, 3, 2)
              -   [[[1 2] [3 4] [5 6]]]
                  [[[7 8] [9 10] [11 12]]]

      -   **GPU나 TPU와 같은 연산 가속을 위한 특수한 하드웨어에서 실행할 수 있다.**
      -   **Backward pass에서 계산된 Gradient (.grad)을 저장한다.**
      -   **기본적으로 torch.Tensor에 어떤 operation (더하기, 곱셈 등)을 취하면 해당 operation이 Computational Graph에 기록된다.**
          -   해당 Tensor와 다른 배열들과 Parameter 간의 경사를 구하는 것.
          -   Auto Differentation
              -   Backward Propagation의 핵심개념이자 작동원리
              -   모두 Reverse Differentiation을 활용해서 Gradient Descent에 필요한 Gradient을 계산하는 것임
              -   즉, w = (w0, w1, ... ,wN)인 어떤 임의의 합성함수 L(w)에 대해서 우리는 dL/dwi을 구할 수 있는 것임.

### [실습] PyTorch 기초 - Dataset과 DataLoader

-   Dataset과 Data Loader
    -   Dataset과 Data Loader은 PyTorch에서 제공하는 추상 클래스
    -   ***Dataset*** (torch.utils.data.Dataset)
        -   Mini-batch를 구성할 각 data sample을 하나씩 불러오는 기능을 수행한다.
    -   ***Dataloader (torch.utils.data.DataLoader)***
        -   Dataset에서 불러온 각 data sample들을 모아서 mini-batch로 구성하는 기능을 수행한다.

    -   그리고 Data Loader를 통해서 Data sample들을 병렬적으로 불러오고, 데이터 셔플 등의 작업을 간단하게 수행할 수 있다.

-   Dataset의 뼈대대
    -   __init__ function
    -   __len__ function
    -   __getitem__ function

    -   예:

```
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self):
        # 데이터셋의 전처리를 수행하는 부분
    def __len__(self):
        # 데이터셋의 길이 (즉, 총 샘플의 갯수를 명시하는 부분)
    def __getitem__(self, item):
        # 데이터셋에서 특정 1개의 샘플을 가져오는 부분
        # 참고로 item은 index
```

-   Data Loader의 Input:
    -   dataset
    -   batch_size: mini-batch의 크기
    -   shuffle (binary): 무작위 순서로 데이터를 샘플링 할 것인지
    -   num_workers: 데이터 로딩에 사룡할 subprocess 개수 (병렬 처리)
    -   pin_memory (binary): GPU memory에 pin 할 것인지
    -   drop_last (binary): 마지막 mini-batch을 drop할 것인지

-   참고로
    -   num_workers 수가 많을수록 데이터 로딩이 더 빠르지만, 그만큼 CPU core 갯수도 충분해야한다. (CPU core 갯수보다 num_workers가 더 많으면 오히려 느려지는 경우 발생)
    -   pin_memory=True로 했을 시, GPU (cuda) memory를 미리 할당, 확보시켜서 조금 더 빠르게 데이터를 GPU에 올릴 수 있다.

-   다음 내용을 Jupyter Notebook에서 살펴보자:
    -   CIFAR10 (torchvision.datasets.CIFAR10)을 활용
    -   torch.utils.Dataset으로 Custom Dataset 구현
    -   torch.utils.DataLoader 활용

### [실습] PyTorch 기초 - Transforms

-   Transforms
    -   Torchvision에서는 Computer Vision에서 사용되는 다양한 모델, 데이터셋들을 제공
    -   여기서 torchvision의 transforms 모듈은 이미지 전처리에 관련된 유용한 기능들을 제공
    -   예를 들어,
        -   ToTensor : image를 tensor로 변환하고 0~1 사이 값으로 normalize
        -   Data augmentation (데이터 증강)
            -   Gaussian Blur
            -   Random Affine, 등

-   PreProcessing(전처리)
    -   Deep Learning Pipeline(process)
        -   Raw Input -> Deep Learning Pipeline -> Cat! (Output)
    -   MLOps에서의 pipeline의미는 조금 다름
    -   여기서는 raw input에 대해서 output으로 mapping하는 일련의 과정을 의미한다. 

    -   CV case.
        -   전처리 -> 딥러닝모델 -> 후처리
        -   [전처리]
            -   이미지를 H' W'로 resize한다.
            -   왜냐하면 NN의 input으로 mini-batch를 구성하는 데이터들은 동일한 dimension (shape)을 가져야 한다.
            -   그래야 mini-batch을 Matrix 혹은 Tensor로 표현할 수 있다!
            -   resized 된 이미지를 numpy 배열로 변환
            -   0~1사이 값으로 normalize 한다
            -   * Torchvision의 Transforms은 CV와 관련된 전처리 함수, 기능들을 포함!

        -   [딥러닝 모델]
            -   NN Model은 전처리 값을 입력받아서 또 다른 Tensor값을 출력한다.
            ```
            model(img)
            tensor([0.9900, 0.100])
            # 고양이일 확률 예측 0.999
            # 강아지일 확률 예측 0.100
            ```
        
        -   [후처리]
            -   Cat!
            -   비록 해당 예제에서는 간단하지만 더 복잡한 전처리의 task도 많이 있다.
            -   e.g. Object Detection, Segmentation, NLP ...

### [실습] PyTorch로 구현해보는 Neural Network
-   PyTorch 모델의 기본 뼈대
    -   __init__ 함수
        -   NN을 구성하고 있는 layer들을 명시하고 initialize한다.
    -   forward 함수
        -   입력에 대한 forward pass을 정의한다.
    ```
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            # Neural Network을 구성하는 layer들을
            # initialize하는 부분
            self.fc_layers = nn.Sequential(
                nn.Linear(784, 784 // 4),
                nn.ReLU(),
                nn.Linear(784 // 4, 784 // 16),
                nn.ReLU(),
                nn.Linear(784 // 16, 10),
                nn.Sigmoid(),
            )

        def forward(self, x):
            # Neural Network의 forward pass을 정의하는 부분
            # x은 input tensor
            x = torch.flatten(x, start_dim=1)
            x = self.fc_layers(x)
            return x
    ```
    -   torchsummary을 사용해서 모델의 summary 뽑기
    
