  
# 문자열을 뒤집는 함수

# 내 방식
answer =[]
s = ['h','e','l','l','o']
def solution(s:str):
    for i in range(len(s)):
        answer.append(s.pop(-1))
    return answer

# print(solution(s))

#  풀이
def reverseString(s):
    s.reverse()
    return s
print(reverseString(s))