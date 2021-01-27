



def sumMatrix(A,B):
    answer = [[c + d for c, d in zip(a, b)] for a, b in zip(A,B)]
    return answer

# 아래는 테스트로 출력해 보기 위한 코드입니다.
print(sumMatrix([[1,2], [2,3]], [[3,4],[5,6]]))

x = [1,2,3]
y = [4,5,6]


answer1 = [x+y for x,y in zip(x,y)]

print(answer1)
