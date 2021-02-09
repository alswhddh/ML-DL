import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

#현재경로 확인
print(os.getcwd())

#데이터 불러오기
boston = pd.read_csv("./실습/part2_data/Boston_house.csv")

#데이터 불러오기 확인
print(boston.head())

#target 제외한 데이터만 뽑기
boston_data = boston.drop(['Target'],axis=1)

#데이터 통계 보기

print(boston_data.describe())

"""
타겟 데이터
1978 보스턴 주택 가격
506개 타운의 주택 가격 중앙값 ( 단위 1,000 달러 )

특징데이터
CRIM : 볌죄율
INDUS : 비소매상업지역 면적 비율
NOX: 일산화질소 농도
RM: 주택당 방 수
LSTAT: 인구 중 하위 계층 비율
B: 인구 중 흑인 비율
PTRATIO: 학생/교사 비율
ZN: 25,000 평방피트를 초과 거주지역 비율
CHAS: 찰스강의 경계에 위차한 경우 1, 아니면 0
AGE: 1940년 이전에 건축된 주택의 비율
RAD: 방사형 고속도로까지의 거리
DIS: 직업센터의 거리
TAX: 재산세율
"""

#CRIM / RM / LSTAT 세개의 변수로 각각 단순 선형 회귀 분석하기

#타겟 설정
target = boston[['Target']]
crim = boston[['CRIM']]
rm = boston[['RM']]
lstat = boston[['LSTAT']]

#crim변수에 상수항추가하기
crim1 = sm.add_constant(crim,has_constant="add")

#sm.OLS 적합시키기
model1 = sm.OLS(target,crim1)
fitted_model1 = model1.fit()

#summary 함수통해 결과 출력
print(fitted_model1.summary())

#회귀 계수 출력
print(fitted_model1.params)

#회귀 계수 x 데이터(x)
np.dot(crim1,fitted_model1.params)
pred1 = fitted_model1.predict()

#직접구한 yhat과 predict 함수를 통해 구한 yhat차이
print(np.dot(crim1,fitted_model1.params) - pred1)

#적합시킨 직선 시각화
import matplotlib.pyplot as plt
plt.yticks(fontname = "Arial")
plt.scatter(crim,target,label="data")
plt.plot(crim,pred1,label="result")
plt.legend()
plt.show()

plt.scatter(target,pred1)
plt.xlabel("real_value")
plt.ylabel("pred_value")
plt.show()

#residual 시각화
fitted_model1.resid.plot()
plt.xlabel("residual_number")
plt.show()

#잔차의 합계산해보기
np.sum(fitted_model1.resid)

#위와 동일하게 rm변수와 lstat 변수로 각각 단순선형회귀분석 결과보기

#상수항추가
rm1 = sm.add_constant(rm,has_constant="add")
lstat1 = sm.add_constant(lstat,has_constant="add")

#회귀모델 적합
model2 = sm.OLS(target,rm1)
fitted_model2 = model2.fit()

model3 = sm.OLS(target,lstat1)
fitted_model3 = model3.fit()

#rm 모델결과출력
fitted_model2.summary()

#lstat 모델결과출력
fitted_model3.summary()

# 각각 yhat_예측하기
pred2 = fitted_model2.predict(rm1)
pred3 = fitted_model3.predict(lstat1)

#rm 시각화
plt.scatter(rm,target,label="data")
plt.plot(rm,pred2,label="result")
plt.legend()
plt.show()

#lstat 시각화
plt.scatter(lstat,target,label="data")
plt.plot(lstat,pred3,label="result")
plt.legend()
plt.show()

#rm모델 residual시각화
fitted_model2.resid.plot()
plt.xlabel("residual_number")
plt.show()

#lstat모델 residual시각화
fitted_model3.resid.plot()
plt.xlabel("residual_number")
plt.show()

# 세 모델의 residual 비교
fitted_model1.resid.plot(label="crim")
fitted_model2.resid.plot(label="rm")
fitted_model3.resid.plot(label="lstat")
plt.legend()
plt.show()
