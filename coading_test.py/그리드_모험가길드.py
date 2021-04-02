n = int(input())
data = list(map(int, input().split()))
data.sort()

result = 0 # 총 그룹 수
count = 0 # 현재 그룹의 모험가 수

for i in data: # 공포도 낮은 것부터 확인
    count += 1 # 현재 그룹에 해당 모험가를 포함시켜라
    if count >= 1: # 현재 그룹에 포함한 모험가의 수가 현재 공포도 이상이라면 그룹결성
        result += 1
        count = 0 # 현재 그룹의 모험가 수 초기화
print(result)