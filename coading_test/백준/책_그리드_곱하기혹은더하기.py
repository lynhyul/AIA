data = input()

# 첫번째 문자를 숫자로 바꾼다
result = int(data([0]))
count = []

for i in range(0, len(data)):
    # 두 수 중에서 하나라도 '0'혹은 '1'인 경우 곱하기 보다는 더하기 추천
    num = int(data[1])
    if num <= 1 or result <= 1: # 두수 num, 결과가 0이거나 1일 때
        result += 1
    else :
        result *= num

print(result)