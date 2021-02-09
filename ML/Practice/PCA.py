import pandas as pd

#데이터 프레임 생성
df = pd.DataFrame(columns=['calory','breakfast','lunch','dinner','exercise','body_shape'])

#데이터 셋
df.loc[0] = [1200,1,0,0,2,'Skinny']
df.loc[1] = [2800,1,1,1,1,'Normal']
df.loc[2] = [3500,2,2,1,0,'Fat']
df.loc[3] = [1400,0,1,0,3,'Skinny']
df.loc[4] = [5000,2,2,2,0,'Fat']
df.loc[5] = [1300,0,0,1,2,'Skinny']
df.loc[6] = [3000,1,0,1,1,'Normal']
df.loc[7] = [4000,2,2,2,0,'Fat']
df.loc[8] = [2600,0,2,0,0,'Normal']
df.loc[9] = [3000,1,2,1,1,'Fat']

print(df.head(10))

x = df[['calory','breakfast','lunch','dinner','exercise']]

print(x.head(10))

y = df[['body_shape']]

print(y.head(10))

from sklearn.preprocessing import StandardScaler

xStd = StandardScaler().fit_transform(x)

print(xStd)

import numpy as np

features = xStd.T

convarianceMatrix = np.cov(features)
print(convarianceMatrix)

eigValue , eigVector = np.linalg.eig(convarianceMatrix)

print('EigenVactors \n%s' %eigVector)

print('\nEigenValues \n%s' %eigValue)

firstPC = eigValue[0] / sum(eigValue)

print(firstPC)

projectedX = xStd.dot(eigVector.T[0])

print(projectedX)

result = pd.DataFrame(projectedX,columns=['PC1'])
result['y-axis'] = 0.0
result['label'] = y

import matplotlib.pyplot as plt
import seaborn as sns

sns.lmplot('PC1','y-axis',data=result,fit_reg=False,
           scatter_kws={"s":50},
           hue="label")

plt.title("PCA result")
plt.show()

"""
#패키지로 PCA 알고리즘 사용하기

from sklearn import decomposition
pca = decomposition.PCA(n_components=1)
sklPcaX = pca.fit_transform(xStd)

sklResult = pd.DataFrame(sklPcaX,columns=['PC1'])
sklResult['y-axis'] = 0.0
sklResult['label'] = y

sns.lmplot('PC1','y-axis',data=sklResult,fit_reg=False,
           scatter_kws={"s":50},
           hue="label")

plt.title("PCA result")
plt.show()
"""