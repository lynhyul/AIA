# 인형뽑기

board = [[0,0,0,0,0],[0,0,1,0,3],[0,2,5,0,1],[4,2,4,4,2],[3,5,1,3,1]]
moves = [1,5,3,5,1,2,1,4]
import numpy as np
def solution(board, moves):
    answer = 0
    basket=[]
    board = np.array(board)
    for i in range(len(moves)):
        temp = board[:,(moves[i]-1)]
        which_nonzero = np.nonzero(temp)[0] # np.nonzero -> list나 배열에서 zero가 아닌 부분의 좌표를 받아온다.
        if(len(which_nonzero)>0):
            basket.append(temp[which_nonzero[0]])
            if(len(basket)>1):
                if(basket[-1]==basket[-2]):
                    basket.pop()
                    basket.pop()
                    answer +=2
            board[which_nonzero[0],(moves[i]-1)]=0
        else :continue
    

    return answer

# print(solution(board, moves))


def solution(board, moves):
    stacklist = []
    answer = 0

    for i in moves:
        for j in range(len(board)):
            if board[j][i-1] != 0:
                stacklist.append(board[j][i-1])
                board[j][i-1] = 0

                if len(stacklist) > 1:
                    if stacklist[-1] == stacklist[-2]:
                        stacklist.pop(-1)
                        stacklist.pop(-1)
                        answer += 2     
                break

    return answer