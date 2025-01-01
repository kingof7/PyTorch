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
