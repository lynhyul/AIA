#문제 출저 : https://programmers.co.kr/learn/courses/30/lessons/72410

# 카카오에 입사한 신입 개발자 네오는 "카카오계정개발팀"에 배치되어, 카카오 서비스에 가입하는 유저들의 아이디를 생성하는 업무를 담당하게 되었습니다. 
# "네오"에게 주어진 첫 업무는 새로 가입하는 유저들이 카카오 아이디 규칙에 맞지 않는 아이디를 입력했을 때, 
# 입력된 아이디와 유사하면서 규칙에 맞는 아이디를 추천해주는 프로그램을 개발하는 것입니다.
# 다음은 카카오 아이디의 규칙입니다.

# 아이디의 길이는 3자 이상 15자 이하여야 합니다.
# 아이디는 알파벳 소문자, 숫자, 빼기(-), 밑줄(_), 마침표(.) 문자만 사용할 수 있습니다.
# 단, 마침표(.)는 처음과 끝에 사용할 수 없으며 또한 연속으로 사용할 수 없습니다.
# "네오"는 다음과 같이 7단계의 순차적인 처리 과정을 통해 신규 유저가 입력한 아이디가 카카오 아이디 규칙에 맞는 지 검사하고 
# 규칙에 맞지 않은 경우 규칙에 맞는 새로운 아이디를 추천해 주려고 합니다.
# 신규 유저가 입력한 아이디가 new_id 라고 한다면,

# 1단계 new_id의 모든 대문자를 대응되는 소문자로 치환합니다.
# 2단계 new_id에서 알파벳 소문자, 숫자, 빼기(-), 밑줄(_), 마침표(.)를 제외한 모든 문자를 제거합니다.
# 3단계 new_id에서 마침표(.)가 2번 이상 연속된 부분을 하나의 마침표(.)로 치환합니다.
# 4단계 new_id에서 마침표(.)가 처음이나 끝에 위치한다면 제거합니다.
# 5단계 new_id가 빈 문자열이라면, new_id에 "a"를 대입합니다.
# 6단계 new_id의 길이가 16자 이상이면, new_id의 첫 15개의 문자를 제외한 나머지 문자들을 모두 제거합니다.
#      만약 제거 후 마침표(.)가 new_id의 끝에 위치한다면 끝에 위치한 마침표(.) 문자를 제거합니다.
# 7단계 new_id의 길이가 2자 이하라면, new_id의 마지막 문자를 new_id의 길이가 3이 될 때까지 반복해서 끝에 붙입니다.
# 예를 들어, new_id 값이 "...!@BaT#*..y.abcdefghijklm" 라면, 위 7단계를 거치고 나면 new_id는 아래와 같이 변경됩니다.

# 1단계 대문자 'B'와 'T'가 소문자 'b'와 't'로 바뀌었습니다.
# "...!@BaT#*..y.abcdefghijklm" → "...!@bat#*..y.abcdefghijklm"

# 2단계 '!', '@', '#', '*' 문자가 제거되었습니다.
# "...!@bat#*..y.abcdefghijklm" → "...bat..y.abcdefghijklm"

# 3단계 '...'와 '..' 가 '.'로 바뀌었습니다.
# "...bat..y.abcdefghijklm" → ".bat.y.abcdefghijklm"

# 4단계 아이디의 처음에 위치한 '.'가 제거되었습니다.
# ".bat.y.abcdefghijklm" → "bat.y.abcdefghijklm"

# 5단계 아이디가 빈 문자열이 아니므로 변화가 없습니다.
# "bat.y.abcdefghijklm" → "bat.y.abcdefghijklm"

# 6단계 아이디의 길이가 16자 이상이므로, 처음 15자를 제외한 나머지 문자들이 제거되었습니다.
# "bat.y.abcdefghijklm" → "bat.y.abcdefghi"

# 7단계 아이디의 길이가 2자 이하가 아니므로 변화가 없습니다.
# "bat.y.abcdefghi" → "bat.y.abcdefghi"

# 따라서 신규 유저가 입력한 new_id가 "...!@BaT#*..y.abcdefghijklm"일 때, 네오의 프로그램이 추천하는 새로운 아이디는 "bat.y.abcdefghi" 입니다.


import re

new_id = "...!@BaT#*..y.abcdefghijklm"


def solution(new_id) :
    # 1단계 소문자로 치환
    new_id = new_id.lower() # 대문자를 소문자로 만든다.

    # 2단계 지정문자를 제외한 특수문자 제거
    answer = ''
    for word in new_id :
        if word.isalnum() or word in '-_.' :
            answer += word
    
    # 3단계 중복된, ....와 같은 것을 .로 줄이기
    while '..' in answer :
        answer = answer.replace('..','.')

    # 4단계 아이디의 처음에 위치한 '.'가 제거
    answer = answer[1:] if answer[0] == '.' and len(answer) > 1 else answer
    answer = answer[:-1] if answer[-1] == '.' else answer


    # 5단계 new_id가 빈 문자열이라면, new_id에 "a"를 대입합니다.
    answer = 'a' if answer == '' else answer
    
    # 6단계 아이디의 길이가 16자 이상이므로, 처음 15자를 제외한 나머지 문자들이 제거
    if len(answer) >= 16 :
        answer = answer[:15]
        if answer[-1] == '.' :
            answer = answer[:-1]
    # 7단계 new_id의 길이가 2자 이하라면, new_id의 마지막 문자를 new_id의 길이가 3이 될 때까지 반복해서 끝에 붙입니다.
    if len(answer) <= 3:
        answer = answer + answer[-1] * (3-len(answer))
    return answer


    # "bat.y.abcdefghi" → "bat.y.abcdefghi"

# 다른 풀이 ===================================================================================
# 정규표현식

import re

# re.sub('패턴', '바꿀문자열', '문자열', 바꿀횟수)

def solution(new_id):
    st = new_id
    st = st.lower() # 1단계
    print(st)
    st = re.sub('[^a-z0-9\-_.]', '', st)    # 2단계
    print(st)   # ...bat..y.abcdefghijklm
    st = re.sub('\.+', '.', st) # 3단계
    print(st)   # .bat.y.abcdefghijklm
    st = re.sub('^[.]|[.]$', '', st)    # 4단계
    print(st)   # bat.y.abcdefghijklm
    st = 'a' if len(st) == 0 else st[:15]   # 5단계
    print(st)   # bat.y.abcdefghi
    st = re.sub('^[.]|[.]$', '', st)    # 6단계
    print(st)   # bat.y.abcdefghi
    st = st if len(st) > 2 else st + "".join([st[-1] for i in range(3-len(st))])    # 7단계
    return st

solution(new_id)