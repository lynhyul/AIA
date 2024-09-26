# 프로그래머스 모바일은 개인정보 보호를 위해 고지서를 보낼 때
# 고객들의 전화번호의 일부를 가립니다.
# 전화번호가 문자열 phone_number로 주어졌을 때, 
# 전화번호의 뒷 4자리를 제외한 나머지 숫자를 전부 *으로 가린 
# 문자열을 리턴하는 함수, solution을 완성해주세요.

# 제한 조건
# s는 길이 4 이상, 20이하인 문자열입니다.

phone_number = "01033334444" # 11글자 / 7글자를 가려야함
phone_number = list(map(int, phone_number))
# [0, 1, 0, 3, 3, 3, 3, 4, 4, 4, 4]
def solution(phone_number):
    if 3 < len(phone_number) < 21 :
        phone_number = list(map(int, phone_number))
        del phone_number[(len(phone_number)-4):-4]
        for i in range(len(phone_number)-4) :
            phone_number.insert(1,'*')
    return phone_number

print(solution(phone_number))