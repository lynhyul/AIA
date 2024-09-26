# 조이스틱으로 알파벳 이름을 완성하세요. 맨 처음엔 A로만 이루어져 있습니다.
# ex) 완성해야 하는 이름이 세 글자면 AAA, 네 글자면 AAAA

# 조이스틱을 각 방향으로 움직이면 아래와 같습니다.

# ▲ - 다음 알파벳
# ▼ - 이전 알파벳 (A에서 아래쪽으로 이동하면 Z로)
# ◀ - 커서를 왼쪽으로 이동 (첫 번째 위치에서 왼쪽으로 이동하면 마지막 문자에 커서)
# ▶ - 커서를 오른쪽으로 이동
# 예를 들어 아래의 방법으로 "JAZ"를 만들 수 있습니다.

# - 첫 번째 위치에서 조이스틱을 위로 9번 조작하여 J를 완성합니다.
# - 조이스틱을 왼쪽으로 1번 조작하여 커서를 마지막 문자 위치로 이동시킵니다.
# - 마지막 위치에서 조이스틱을 아래로 1번 조작하여 Z를 완성합니다.
# 따라서 11번 이동시켜 "JAZ"를 만들 수 있고, 이때가 최소 이동입니다.
# 만들고자 하는 이름 name이 매개변수로 주어질 때, 이름에 대해 조이스틱 조작 횟수의 최솟값을 return 하도록 solution 함수를 만드세요.

# 제한 사항
# name은 알파벳 대문자로만 이루어져 있습니다.
# name의 길이는 1 이상 20 이하입니다.
# 입출력 예
# name	return
# "JEROEN"	56
# "JAN"     23


# 1. 일단 방향키를 구현한다.
# ▲ - 다음 알파벳
# ▼ - 이전 알파벳 (A에서 아래쪽으로 이동하면 Z로)
# ◀ - 커서를 왼쪽으로 이동 (첫 번째 위치에서 왼쪽으로 이동하면 마지막 문자에 커서)
# ▶ - 커서를 오른쪽으로 이동
# 예를 들어 아래의 방법으로 "JAZ"를 만들 수 있습니다.

name = 'JEROEN'


def solution(name):
    joy = []
    answer = 0
    #배정
    for i in range(len(name)):
        if name[i]=='A':
            continue
        else:
            joy.append(i)
        temp = ord(name[i])-ord('A')
        print(temp) # 8 4 17 14 4 13
        if temp>13: # 최선의 방법을 찾는것이기 때문에 26개의 문자중 1/2값인 13을 넘기게 되면 반대로 탐색
            answer += 26-temp
            # print(answer)   # 22, 34 
        else:
            answer += temp
            # print(answer)   # 9, 13 ,38, 51
    #이제 index 기반으로 다음 위치 그리디하게 찾기
    current = 0
    for i in range(len(joy)):
        move_list = [abs(x-current) if abs(x-current)<=len(name)/2 else len(name)-abs(x-current) for x in joy]
        print(move_list)
        answer += min(move_list)
        current = joy.pop(move_list.index(min(move_list)))
    return answer

solution(name)