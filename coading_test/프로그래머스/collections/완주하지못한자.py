# 완주하지 못한 자

participant = ["leo", "kiki", "eden"]
completion = ["eden", "kiki"]

# 실패작 -> 중복 이름도 제거해버림
# def solution(participant, completion):
#     answer = ''
#     for i in participant:
#         if i not in completion:
#             answer = i
#     return answer

# print(solution(participant, completion))

# Counter 사용 -> collection.Counter는 {자료 이름 : 갯수} 형태로 나온다 객체들끼리 뺄셈도 가능
    # print(collections.Counter(participant)) # Counter({'leo': 1, 'kiki': 1, 'eden': 1})
    # print(collections.Counter(completion))  # Counter({'eden': 1, 'kiki': 1})
import collections

def solution(participant, completion):
    answer = collections.Counter(participant) - collections.Counter(completion)
    print(answer.keys())   # Counter({'leo': 1}) / dict_keys(['leo'])
    return list(answer.keys())[0]
print(solution(participant,completion)) # leo


# p = ['mislav', 'stanko', 'mislav', 'ana']
# c = ['stanko', 'ana', 'mislav']

# print(collections.Counter(p))
# # Counter({'mislav': 2, 'stanko': 1, 'ana': 1})

# print(collections.Counter(c))
# # Counter({'stanko': 1, 'ana': 1, 'mislav': 1}


# hash()를 이용해서 주소값으로 변환 후 원소 하나하나를 제거
def solution(participant, completion):
    answer = ''
    temp = 0
    dic = {}
    for part in participant:
        dic[hash(part)] = part
        temp += int(hash(part))
    for com in completion:
        temp -= hash(com)
    answer = dic[temp]

    return answer