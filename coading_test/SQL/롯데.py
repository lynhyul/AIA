# n = 0 -> '0'
# n = 1 -> '01'
# n = 2 -> '0110'
# n = 3 -> '01101001' 이런식으로 반복되는것이 있다.
# a, b라는 추가 변수가 있는데, n=3일때 나오는 값인 '01101001'에서 a는 시작점, b는 끝점
# 예를들어 a가 2, b가 6이면 answer는 '11010'이 된다.
# 이를 구현하라


# def solution(n) :
n = 3
answer = '0'
for i in range(n) :
    for j in answer :
        if j == '0' :  
            answer += '1'
        if j == '1' :
            answer += '0'   
print(answer)
# 0110100110010110