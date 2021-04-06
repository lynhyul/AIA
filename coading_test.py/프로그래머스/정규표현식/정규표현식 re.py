a = 'abcdef\n' # 엔터역할
b = r'abcdef\n' #문자 그대로 출력
print(b) # abcdef\n

### 정규 표현식

# \w - 문자 character와 일치[a-z A-Z 0-9_] 대.소문자, 숫자
# \W non alpha-numeric
# \s 공백문자와 일치
# \t, \n, \r => tab, newline, return
# \d = 숫자 character와 일치 [0-9] 0~9까지 숫자 1개를 의미
# ^= 시작, $ = 끝 각각 문자열의 시작과 끝을 의미
# \가 붙으면 스페셜한 의미가 사라짐.

### search metod
import re
# m = re.search(r'abc','123abcdef') # abcdef에서 abc라는 글자를 찾는다.
# print(m)    # <re.Match object; span=(3, 6), match='abc'>
#             # 없을경우 None을 반환
# print(m.start())  #3 / abc가 포함된 인덱스가 3부터 시작
# print(m.end())  # 6  / 6부터 포함되지 않음
# print(m.group())    # abc

m = re.search(r'\d\d\d', '112abcdef119') # 숫자 3개가 연달아 오는것
print(m)    # <re.Match object; span=(0, 3), match='112'>

m1 = re.search(r'..\w\w', '^%^%#$$$$ABCDabcd')
print(m1)   # <re.Match object; span=(7, 11), match='$$AB'>

# 문자들의 범위를 나타내기 위해 사용

# [ ] 내부의 메타 캐릭터는 캐릭터 자체를 나타냄
# e.g)
# [abck] : a or b or c or k
# [abc.^] : a or b or c or . or ^
# [a-d] : -와 함께 사용되면 해당 문자 사이의 범위에 속하는 문자중
# [0-9] : 모든 숫자
# [a-z] 모든 소문자
# [A-Z] : 모든 대문자
# [a-z A-Z 0-9] : 모든 알파벳 문자 및 숫자
# [^0-9] : ^가 맨 앞에 사용 되는 경우 해당 문자 패턴이 아닌 것과 매칭

print(re.search(r'[cbm]at', 'cat'))
# <re.Match object; span=(0, 3), match='cat'>

print(re.search(r'[0-8]haha', '7hahah'))
# <re.Match object; span=(0, 5), match='7haha'>
print(re.search(r'[abc.^]aron', 'daron'))   # None
print(re.search('r[abc.^]aron', '0aron'))   # None

print(re.search(r'\Sand', 'apple land banana')) 
#<re.Match object; span=(6, 10), match='land'>
print(re.search(r'\.and', '.and')) 
#<re.Match object; span=(0, 4), match='.and'>
# \, \\ 메타캐릭터가 캐릭터 자체를 표현하도록 할 경우 사용

print(re.search(r'p.g', 'pig'))

print(re.search(r'a[bcd]*b', 'abcdbdccb'))
# <re.Match object; span=(0, 9), match='abcdbdccb'>

# '+' -> 1번 이상의 패턴이 발생
# '*' -> 0번 이상의 패턴이 발생 / 최대한 많이 발생 시키려함
# '?' -> 0 혹은 1번의 패턴이 발생

print(re.search(r'b\w+a', 'banana'))
# <re.Match object; span=(0, 6), match='banana'>

print(re.search(r'i+', 'piigiii'))
# <re.Match object; span=(1, 3), match='ii'>

print(re.search(r'pi+g', 'pg')) # None 

print(re.search(r'pi*g', 'pg'))
# <re.Match object; span=(0, 2), match='pg'>

print(re.search(r'https?', 'http://www.naver.com'))
# <re.Match object; span=(0, 4), match='http'>

print(re.search(r'b\w+a', 'cabana'))
# <re.Match object; span=(2, 6), match='bana'>

print(re.search(r'^b\w+a', 'cabana'))   # None
print(re.search(r'^b\w+a', 'babana'))
# <re.Match object; span=(0, 6), match='babana'>

print(re.search(r'b\w+a$', 'cabana')) # a로 끝나는 패턴 추출
# <re.Match object; span=(2, 6), match='bana'>

print(re.search(r'b\w+a$', 'cabanp')) # None

# group을 쓰면 re를 통해 반환 된 값을 str로 변형이 가능합니다!! 중요오!!!
m2 = re.search(r'(\w+)@(.+)', 'test@gmail.com')
print(m2.group())   # test@gmail.com
print(m2.group(1))  # test
print(m2.group(2))  # gmail.com

# 반복의 횟수 명시
print(re.search('pi{3}g', 'piiiig')) # None
print(re.search('pi{3}g', 'piiig'))
# <re.Match object; span=(0, 5), match='piiig'>
print(re.search('pi{3,5}g', 'piiiiig')) # {최소, 최대}
# <re.Match object; span=(0, 7), match='piiiiig'>

# 미니멈 매칭
print(re.search(r'<.+>', '<html>haha</html>'))
# <re.Match object; span=(0, 17), match='<html>haha</html>'>
print(re.search(r'<.+?>', '<html>haha</html>')) 
# ? 를 추가함으로써 패턴에 포함된것중 최소한의 길이로만 찾는다.
# <re.Match object; span=(0, 6), match='<html>'>

print(re.search(r'a{3,5}', 'aaaaa'))
# <re.Match object; span=(0, 5), match='aaaaa'>
print(re.search(r'a{3,5}?', 'aaaaa'))
# <re.Match object; span=(0, 3), match='aaa'>

# List<Map<String, Object>>로 변환가능
