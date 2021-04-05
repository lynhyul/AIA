# 두 수의 최소공배수(Least Common Multiple)란 입력된 두 수의 배수 중 공통이 되는 가장 작은 숫자를 의미합니다.
#  예를 들어 2와 7의 최소공배수는 14가 됩니다. 
#  정의를 확장해서, n개의 수의 최소공배수는 n 개의 수들의 배수 중 공통이 되는 가장 작은 숫자가 됩니다. 
#  n개의 숫자를 담은 배열 arr이 입력되었을 때 이 수들의 최소공배수를 반환하는 함수, solution을 완성해 주세요.

# 제한 사항
# arr은 길이 1이상, 15이하인 배열입니다.
# arr의 원소는 100 이하인 자연수입니다.
# 입출력 예
# arr	result
# [2,6,8,14]	168
# [1,2,3]	6

from math import gcd
import math

arr = [2,6,8,14]


# 최대 공약수 : math의 함수에서 gcd를 사용 -> gcd(x,y)

# 최소공배수
def lcm(x, y) :
    return x*y // gcd(x,y)

print(lcm(2,5)) # 10
# print(arr.pop())    # 14 / 리스트의 가장 마지막 요소를 제거 한 뒤 그 값으로 반환한다.

# N개의 최소공배수 (이 문제에서 요구하는 방법)

def solution(arr):
    while True :
        arr.append(lcm(arr.pop(), arr.pop()))
        if len(arr) == 1 :
            return arr[0]

print(solution(arr))

# =========================================================================================
# 모범 풀이

from fractions import gcd


def nlcm(arr):      
    answer = arr[0]     # 없으면 오류뜨는데 무슨 역할인지는 모름
    for n in arr:
        answer = (n * answer) // math.gcd(n, answer)

    return answer

# 아래는 테스트로 출력해 보기 위한 코드입니다.
print(nlcm(arr))    # 168.0