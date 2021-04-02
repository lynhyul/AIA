# LZW 압축은 다음 과정을 거친다.

# 길이가 1인 모든 단어를 포함하도록 사전을 초기화한다.
# 사전에서 현재 입력과 일치하는 가장 긴 문자열 w를 찾는다.
# w에 해당하는 사전의 색인 번호를 출력하고, 입력에서 w를 제거한다.
# 입력에서 처리되지 않은 다음 글자가 남아있다면(c), w+c에 해당하는 단어를 사전에 등록한다.
# 단계 2로 돌아간다.
# 압축 알고리즘이 영문 대문자만 처리한다고 할 때, 사전은 다음과 같이 초기화된다. 
# 사전의 색인 번호는 정수값으로 주어지며, 1부터 시작한다고 하자.

# 색인 번호   1   2   3   ...   24   25   26
# 단어   A   B   C   ...   X   Y   Z
# 예를 들어 입력으로 KAKAO가 들어온다고 하자.

# 현재 사전에는 KAKAO의 첫 글자 K는 등록되어 있으나, 두 번째 글자까지인 KA는 없으므로, 
# 첫 글자 K에 해당하는 색인 번호 11을 출력하고, 다음 글자인 A를 포함한 KA를 사전에 27 번째로 등록한다.
# 두 번째 글자 A는 사전에 있으나, 세 번째 글자까지인 AK는 사전에 없으므로,
# A의 색인 번호 1을 출력하고, AK를 사전에 28 번째로 등록한다.
# 세 번째 글자에서 시작하는 KA가 사전에 있으므로, KA에 해당하는 색인 번호 27을 출력하고, 
# 다음 글자 O를 포함한 KAO를 29 번째로 등록한다.
# 마지막으로 처리되지 않은 글자 O에 해당하는 색인 번호 15를 출력한다.
# 현재 입력(w)   다음 글자(c)   출력   사전 추가(w+c)
# K   A   11   27: KA
# A   K   1   28: AK
# KA   O   27   29: KAO
# O      15   
# 이 과정을 거쳐 다섯 글자의 문장 KAKAO가 4개의 색인 번호 [11, 1, 27, 15]로 압축된다.

# 입력으로 TOBEORNOTTOBEORTOBEORNOT가 들어오면 다음과 같이 압축이 진행된다.

# 현재 입력(w)   다음 글자(c)   출력   사전 추가(w+c)
# T   O   20   27: TO
# O   B   15   28: OB
# B   E   2   29: BE
# E   O   5   30: EO
# O   R   15   31: OR
# R   N   18   32: RN
# N   O   14   33: NO
# O   T   15   34: OT
# T   T   20   35: TT
# TO   B   27   36: TOB
# BE   O   29   37: BEO
# OR   T   31   38: ORT
# TOB   E   36   39: TOBE
# EO   R   30   40: EOR
# RN   O   32   41: RNO
# OT      34   
# 입력 형식
# 입력으로 영문 대문자로만 이뤄진 문자열 msg가 주어진다. msg의 길이는 1 글자 이상, 1000 글자 이하이다.

# 출력 형식
# 주어진 문자열을 압축한 후의 사전 색인 번호를 배열로 출력하라.

# 입출력 예제
# msg   answer
# KAKAO   [11, 1, 27, 15]
# TOBEORNOTTOBEORTOBEORNOT   [20, 15, 2, 5, 15, 18, 14, 15, 20, 27, 29, 31, 36, 30, 32, 34]
# ABABABABABABABAB   [1, 2, 27, 29, 28, 31, 30]


def solution(msg) :
    table = dict()
    for idx, value in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ",1):
        table[value] = idx
        last_idx = idx
    print(last_idx) # 26
    print(table)
    idx = 1
    answer = []
    letter = msg[0]
    while idx < len(msg):
        # 현재 입력 + 다음 글자 조합이 색인에 없는 경우
        if letter + msg[idx] not in table:
            # 현재 입력을 배열에 저장
            answer.append(table[letter])
            # 새 단어의 색인 값 저장
            last_idx += 1
            table[letter +msg[idx]] = last_idx
            # 현재 입력 = '다음 단어'로 저장
            letter = msg[idx]
            # 다음 단어
            idx += 1
            continue
        #이미 색인되어 있는 단어의 경우, 다음 단어를 현재 입력에 추가
        letter += msg[idx]
        idx += 1
    answer.append(table[letter])
    return answer

print(solution("KAKAO"))