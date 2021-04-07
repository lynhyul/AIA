# 알파벳 대문자와 숫자(0~9)로만 이루어진 문자열이 주어진다.

# 이 문자열을 알파벳 오름차순으로 정렬하고, 숫자는 모두 더해서 뒤에 붙여라.


# [input]
# K1KA5CB7

# [output]
# ABCKK13


word = 'K1KA5CB7'

alphabet = []
num_sum = 0
for i in word:
    if i.isalpha():
        alphabet += [i]
        alphabet.sort()
    elif i.isdigit():
        num_sum += int(i)

print(''.join(alphabet), num_sum, sep='')