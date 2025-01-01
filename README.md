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