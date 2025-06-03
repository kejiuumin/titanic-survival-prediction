# 🚢 타이타닉 생존자 예측 프로젝트 (Titanic Survival Prediction Project)

## 💡 개요 (Overview)
본 프로젝트는 1912년 타이타닉호 침몰 사고 당시의 승객 데이터를 분석하여, 승객의 다양한 특성(성별, 객실 등급, 나이 등)이 생존 여부에 미치는 영향을 탐색하고, 이를 기반으로 생존자를 예측하는 머신러닝 모델을 구축합니다.

## 📊 사용 데이터 (Data Source)
* **데이터셋:** Kaggle - [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)
* `train.csv`: 모델 학습에 사용된 훈련 데이터 (891개 행)
* `test.csv`: 예측을 위한 테스트 데이터 (418개 행)

## 🛠️ 사용 기술 (Technologies Used)
* **Python**
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
* **Environment:** Jupyter Notebook (or VS Code with Jupyter extension)

## 🚀 프로젝트 단계 (Project Steps)

1.  **데이터 로드 및 초기 탐색:**
    * `train.csv`와 `test.csv` 파일 로드.
    * `info()`, `head()` 등을 통해 데이터 구조, 컬럼 타입, 결측치 여부 확인. (`Age`, `Cabin`, `Embarked` 등 결측치 확인)

2.  **탐색적 데이터 분석 (EDA):**
    * 생존자/사망자 비율 분석.
    * **성별(Sex)과 생존:** 여성이 남성보다 압도적으로 높은 생존율을 보임.
    * **객실 등급(Pclass)과 생존:** 1등석 승객의 생존율이 가장 높았으며, 3등석 승객의 생존율이 가장 낮았음.
    * **나이(Age)와 생존:** 어린 아이들의 생존율이 상대적으로 높았음.
    * **승선항(Embarked)과 생존:** 특정 승선항(C) 출신 승객의 생존율이 높았음.

3.  **데이터 전처리 (Data Preprocessing):**
    * **결측치 처리:**
        * `Age`: 이름에서 추출한 호칭(`Mr.`, `Mrs.`, `Miss` 등)별 **중앙값**으로 대체.
        * `Embarked`: 최빈값('S')으로 대체.
        * `Fare` (test set): 중앙값으로 대체.
    * **범주형 변수 인코딩:**
        * `Sex`: `male`은 0, `female`은 1로 레이블 인코딩.
        * `Embarked`: `S`, `C`, `Q`를 0, 1, 2로 레이블 인코딩.
    * **불필요한 컬럼 제거:** `Name`, `Ticket`, `Cabin`, `PassengerId` (train set), `Title` 컬럼 제거.

4.  **머신러닝 모델 훈련 및 평가 (Model Training & Evaluation):**
    * 훈련 데이터를 훈련 세트(80%)와 검증 세트(20%)로 분리.
    * **로지스틱 회귀 (Logistic Regression)** 모델 훈련 및 평가:
        * 정확도: XX.XX%
        * 혼동 행렬 및 분류 보고서 확인.
    * **결정 트리 (Decision Tree)** 모델 훈련 및 평가:
        * 정확도: YY.YY%
        * 혼동 행렬 및 분류 보고서 확인.
    * (선택 사항: 두 모델 성능 비교 및 더 나은 모델 선택)

## 🔑 주요 결과 및 인사이트 (Key Findings & Insights)
* **성별, 객실 등급, 나이**는 타이타닉호 승객의 생존에 매우 중요한 영향을 미쳤습니다. 특히 여성과 고등급 객실 승객의 생존율이 월등히 높았습니다.
* 데이터 전처리 과정을 통해 결측치를 효과적으로 처리하고, 범주형 데이터를 모델이 학습할 수 있는 형태로 변환하는 중요성을 체감했습니다.
* 로지스틱 회귀와 결정 트리 모델을 통해 이진 분류 문제를 해결하고, 모델의 정확도, 정밀도, 재현율 등 다양한 평가 지표를 이해할 수 있었습니다.

## 💡 결론 및 향후 개선점 (Conclusion & Future Work)
이 프로젝트를 통해 데이터 분석의 전반적인 파이프라인(EDA, 전처리, 모델링, 평가)을 직접 경험하고 기본적인 머신러닝 모델을 활용하는 능력을 길렀습니다. 향후 더 높은 예측 정확도를 위해 다음과 같은 개선을 시도해 볼 수 있습니다:
* 더 다양한 **특성 공학**: `FamilySize`, `IsAlone` 등 새로운 유의미한 특성 생성.
* 다른 머신러닝 모델 적용: 랜덤 포레스트(Random Forest), Gradient Boosting 등 앙상블 모델.
* 하이퍼파라미터 튜닝: Grid Search, Random Search 등을 통한 모델 성능 최적화.
* 교차 검증(Cross-validation) 적용: 모델의 일반화 성능 향상.

## 🚀 실행 방법 (How to Run)
1.  **데이터 다운로드:** Kaggle [Titanic Competition 페이지](https://www.kaggle.com/c/titanic/data)에서 `train.csv`와 `test.csv` 파일을 다운로드하여 프로젝트 폴더에 저장합니다.
2.  **필수 라이브러리 설치:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```
3.  **Jupyter Notebook 실행:**
    프로젝트 폴더에서 터미널을 열고 다음 명령어를 입력합니다:
    ```bash
    jupyter notebook
    ```
    또는 VS Code에서 `.ipynb` 파일을 직접 엽니다.
4.  `titanic_analysis.ipynb` 파일을 열고 각 셀을 순서대로 실행하며 분석 과정을 따라갑니다.
