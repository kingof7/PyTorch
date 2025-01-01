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
    - $ conda create --name py312_windows python=3.12 --yes
    - $ conda activate py312_windows
    - $ conda install pytorch torchvision torchaudio -c pytorch --yes
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



