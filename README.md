[추가]geometry.py
geometry.py (Feature Geometry / Neural Collapse 측정)

geometry.py는 이미지 분류 모델의 penultimate layer(feature vector) 를 ID 데이터(학습에 사용한 분포)에서 추출하여, optimizer(SGD vs SAM) 변화에 따른 feature space 기하구조 차이를 정량화하는 스크립트입니다.

본 스크립트는 다음 질문에 답하기 위해 설계되었습니다:

SAM/SGD가 penultimate feature의 구조(분산, 클래스 분리, 저차원화/이방성 등)를 어떻게 바꾸는가?

그 변화가 거리 기반 OOD 탐지(Mahalanobis, kNN, DDU 등) 에 영향을 줄 수 있는 조건을 어떻게 설명할 수 있는가?

무엇을 측정하나? (지표 a–h)

geometry.py는 ID 데이터에서 penultimate feature를 얻고 아래 지표들을 계산합니다.

Neural Collapse 기반 지표

NC1: 클래스 내부 분산(Within-class scatter)이 클래스 평균 구조(Between-class scatter)에 비해 얼마나 남아 있는지(“collapse 정도”를 보는 핵심 지표)

NC2: 마지막 선형 분류기 가중치들이 ETF(정규 심플렉스) 구조에 얼마나 가까운지

NC3: 분류기 가중치와 클래스 평균 방향이 NC 이론에서 말하는 정렬(자기쌍대) 구조를 띄는지

추가 feature geometry 지표

inter_class_mean_dist: 클래스 평균들 사이의 평균 거리(클래스 분리 정도)

anisotropy_mean: 분산이 특정 축에 얼마나 몰려 있는지(이방성/축 집중도; 클래스별 평균)

eff_rank: feature 분산이 얼마나 많은 축으로 퍼져 있는지(유효 차원 수; 클래스별 평균)

(참고) 코드 내부에서는 scatter 행렬(Within/Between scatter) 자체도 계산하지만, JSON에는 용량/가독성을 위해 주로 요약값(예: trace, Frobenius norm)과 최종 지표만 저장합니다.

왜 ID-train에서 측정하나?

Neural Collapse는 일반적으로 학습 목적함수가 충분히 최적화된 뒤, 학습 데이터 분포(ID-train)에서 가장 뚜렷하게 관측되는 현상으로 알려져 있습니다.
또한 optimizer 비교 관점에서 OOD 데이터를 섞으면 “분포 변화 효과”가 함께 섞여 해석이 흔들릴 수 있으므로, 본 스크립트는 동일한 ID 데이터(학습 분포)에서만 feature를 측정해 optimizer 효과를 최대한 공정하게 비교합니다.

실행 흐름(Flow)

geometry.py는 아래 순서로 동작합니다.

체크포인트 로드
*.model 파일을 로드합니다. (state_dict-only 형식 지원)

학습과 동일한 모델 구성
train.py에서 사용한 모델 생성 방식과 동일하게 모델을 구성합니다.
(모델 타입, SN 여부, coeff/mod 등의 인자 포함)

ID 데이터 로더 구성
학습에 사용했던 ID 데이터셋에서 loader를 구성합니다.
필요 시 augmentation 옵션을 학습 설정과 맞출 수 있습니다.

penultimate feature 추출
마지막 선형층(fc)에 hook을 걸어, fc에 입력되는 텐서를 penultimate feature로 수집합니다.

충분통계량 누적(메모리 절약)
전체 feature를 저장하지 않고, 클래스별/전역 통계량을 누적하여 평균/공분산/지표를 계산합니다.

지표 계산 및 JSON 저장
계산 결과를 geometry_results/ 폴더에 JSON으로 저장합니다.

폴더가 없으면 자동 생성

파일명에 주요 인자(dataset/model/sn/coeff/seed 등)를 포함

동일 이름 충돌 시 _v2, _v3 …로 자동 증가하여 덮어쓰지 않음

출력 파일(JSON)

JSON에는 다음이 포함됩니다.

실행 메타 정보: dataset/model/SN/coeff/mod/seed/checkpoint 경로 등

지표 결과: nc1, nc2, nc3, inter_class_mean_dist, anisotropy_mean, eff_rank

scatter 요약: within/between scatter의 trace/Frobenius norm 등

참고 자료(정의/수식/구현 출처)
Neural Collapse 지표(NC1–NC3) 구현 참고

본 레포의 NC 지표 계산 방식은 아래 논문/공식 구현을 참고했습니다.

A Geometric Analysis of Neural Collapse with Unconstrained Features
(공식 GitHub 구현) https://github.com/tding1/Neural-Collapse

validate_NC.py에서 penultimate feature 추출 방식(hook)과 NC1/NC2/NC3 계산 흐름을 참고

Effective rank / Anisotropy 정의 참고

effective rank(유효 차원)과 anisotropy(축 집중도)는 feature geometry 분석에서 널리 쓰이는 표준적 정의를 사용하며, 본 레포에서는 해당 정의를 직접 구현합니다.

자세한 정의/수식은 내부 문서 또는 추후 공유 문서(geometry 문서 링크)에 정리할 예정입니다.
