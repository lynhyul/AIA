# 모든 유저는 [유저 아이디]로 구분한다.

# [유저 아이디] 사용자가 [닉네임]으로 채팅방에 입장 - 
# "Enter [유저 아이디] [닉네임]" (ex. "Enter uid1234 Muzi")

# [유저 아이디] 사용자가 채팅방에서 퇴장 - "Leave [유저 아이디]" (ex. "Leave uid1234")
# [유저 아이디] 사용자가 닉네임을 [닉네임]으로 변경 - 
# "Change [유저 아이디] [닉네임]" (ex. "Change uid1234 Muzi")

#  첫 단어는 Enter, Leave, Change 중 하나이다.
# 각 단어는 공백으로 구분되어 있으며, 알파벳 대문자, 소문자, 숫자로만 이루어져있다.

# 유저 아이디와 닉네임은 알파벳 대문자, 소문자를 구별한다.
# 유저 아이디와 닉네임의 길이는 1 이상 10 이하이다.
# 채팅방에서 나간 유저가 닉네임을 변경하는 등 잘못 된 입력은 주어지지 않는다.


# 입출력 예
# record   
# ["Enter uid1234 Muzi", "Enter uid4567 Prodo","Leave uid1234",
# "Enter uid1234 Prodo","Change uid4567 Ryan"]   
# result
# ["Prodo님이 들어왔습니다.", 
# "Ryan님이 들어왔습니다.", "Prodo님이 나갔습니다.", "Prodo님이 들어왔습니다."]
# 입출력 예 설명
# 입출력 예 #1
# 문제의 설명과 같다.

# =====================================================================

record =  ["Enter uid1234 Muzi", "Enter uid4567 Prodo","Leave uid1234",
"Enter uid1234 Prodo","Change uid4567 Ryan"]

def solution(record):
    answer = []
    namespace = {}
    printer = {'Enter':'님이 들어왔습니다.', 'Leave':'님이 나갔습니다.'}
    for r in record:
        rr = r.split(' ')
        if rr[0] in ['Enter', 'Change']:
            namespace[rr[1]] = rr[2]
            print(namespace)
    for r in record:
        if r.split(' ')[0] != 'Change':
            answer.append(namespace[r.split(' ')[1]] + printer[r.split(' ')[0]])

    return answer

print(solution(record))