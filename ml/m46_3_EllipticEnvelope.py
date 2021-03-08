#가우스 분포

from sklearn.covariance import EllipticEnvelope
import numpy as np

aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100]])
aaa = np.transpose(aaa)
print(aaa.shape)

outlier = EllipticEnvelope(contamination=.3) # 내가 설정한%만큼 (.3 = 30%) 이내의 아웃라이어를 찾아내도록 설정
outlier.fit(aaa)


print(outlier.predict(aaa)) # [ 1  1  1  1 -1  1  1 -1  1 -1]