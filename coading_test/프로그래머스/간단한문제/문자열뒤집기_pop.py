  
# 문자열을 뒤집는 함수
# pop 함수에 대해서도 정리 해봤다.

s1 = ['h','e','l','l','o']

print(s1.pop(-1))   # o
print(s1.pop()) # o


# s1.pop()이나 s1.pop(-1)이나 결과가 같았다. 둘 다 끝에서부터 제거해나갔다.
# 앞에서부터 제거 하려면 for문을 써서 pop(i)로 제거하는 방법도 있겠다.


# 내 방식
answer =[]
s = ['h','e','l','l','o']
def solution(s:str):
    for i in range(len(s)):
        answer.append(s.pop(-1))
    return answer

# print(solution(s))

#  풀이
def reverseString(s):       # 사실 이게 제일 정답인듯하다.
    s.reverse()
    return s
print(reverseString(s))