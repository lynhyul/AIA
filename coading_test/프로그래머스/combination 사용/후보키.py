# 예를 설명하면, 학생의 인적사항 릴레이션에서 모든 학생은 각자 유일한 "학번"을 가지고 있다. 따라서 "학번"은 릴레이션의 후보 키가 될 수 있다.
# 그다음 "이름"에 대해서는 같은 이름("apeach")을 사용하는 학생이 있기 때문에, "이름"은 후보 키가 될 수 없다. 
# 그러나, 만약 ["이름", "전공"]을 함께 사용한다면 릴레이션의 모든 튜플을 유일하게 식별 가능하므로 후보 키가 될 수 있게 된다.
# 물론 ["이름", "전공", "학년"]을 함께 사용해도 릴레이션의 모든 튜플을 유일하게 식별할 수 있지만, 최소성을 만족하지 못하기 때문에 후보 키가 될 수 없다.
# 따라서, 위의 학생 인적사항의 후보키는 "학번", ["이름", "전공"] 두 개가 된다.

# 릴레이션을 나타내는 문자열 배열 relation이 매개변수로 주어질 때, 이 릴레이션에서 후보 키의 개수를 return 하도록 solution 함수를 완성하라.

# 제한사항
# relation은 2차원 문자열 배열이다.
# relation의 컬럼(column)의 길이는 1 이상 8 이하이며, 각각의 컬럼은 릴레이션의 속성을 나타낸다.
# relation의 로우(row)의 길이는 1 이상 20 이하이며, 각각의 로우는 릴레이션의 튜플을 나타낸다.
# relation의 모든 문자열의 길이는 1 이상 8 이하이며, 알파벳 소문자와 숫자로만 이루어져 있다.
# relation의 모든 튜플은 유일하게 식별 가능하다.(즉, 중복되는 튜플은 없다.)

# 입출력 예
# relation
# [["100","ryan","music","2"],["200","apeach","math","2"],["300","tube","computer","3"],["400","con","computer","4"],
# ["500","muzi","music","3"],["600","apeach","music","2"]]
#result
# 2

#=================================================================================================================================

relation = [["100","ryan","music","2"],["200","apeach","math","2"],["300","tube","computer","3"],["400","con","computer","4"],
            ["500","muzi","music","3"],["600","apeach","music","2"]]

# 해당 문제는 풀이에 실패했고, 다른 사람의 풀이를 설명드리겠습니다. 
# 아이디어 : 모든 경우의 수를 만든 후에 유일성, 최소성을 만족하지 않는 부분은 제거
# candidates : combinations을 통해서 가능한 모든 경우의 수를 생성
# final : 가능한 모든 경우의 수에서 유일성을 만족하는 지 확인 
# 튜플 형태로 해당하는 값을 추출해서 길이가 맞는 지 확인합니다. 
# 예) (100, 200, ... , 600) 은 길이가 6으로 유일성 만족 
# answer : 최소성을 만족하는 부분만 추출 
# intersection을 통해서 겹치는 변수가 원본 변수가 같은게 있는 지 확인 

from collections import deque
from itertools import combinations

def solution(relation):
    n_row = len(relation)
    print(n_row)    # 6
    n_col = len(relation[0])  
    print(n_col)    # 4
    candidates=[]
    for i in range(1,n_col+1):
        candidates.extend(combinations(range(n_col),i))
    print(candidates)   
    # [(0,), (1,), (2,), (3,), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 2, 3)]
    final=[]
    for keys in candidates:
        tmp=[tuple([item[key] for key in keys]) for item in relation]
        if len(set(tmp))==n_row:
            final.append(keys)

    answer=set(final[:])
    for i in range(len(final)):
        for j in range(i+1,len(final)):
            if len(final[i])==len(set(final[i]).intersection(set(final[j]))):
                answer.discard(final[j])
                
    return len(answer)

print(solution(relation))   # 2