#가우스 분포

from sklearn.covariance import EllipticEnvelope
import numpy as np

aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],[1000,2000,3,4000,5000,7000,8,9000,10000,1001]])
# aaa = np.transpose(aaa) # 10,2
# print(aaa)

text = []

for out in aaa :
    out = np.transpose([out])
    outlier = EllipticEnvelope(contamination=.2) # 내가 설정한%만큼 (.2 = 20%) 이내의 아웃라이어를 찾아내도록 설정
    outlier.fit(out)
    result = outlier.predict(out) # [ 1  1  1  1 -1  1  1 -1  1  1]
    text.append(result)

print(text)
## 이 기능에서 행은 연대로 간다. 2개의 열을 따로 하고 싶다면 나눠서 해야한다.
## [array([ 1,  1,  1,  1, -1,  1,  1, -1,  1,  1]), array([ 1,  1,  1,  1,  1,  1,  1, -1, -1,  1])]