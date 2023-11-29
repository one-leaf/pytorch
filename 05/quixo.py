# QUIXO 游戏
import numpy as np
import numba
from numba import njit
from numba.typed import Dict

# 初始化游戏板
width = 5
height = 5

# 玩家显示棋子
players = {-1:"X",1:"O"}
forwards = ["w","s","a","d"]
forwards_len = len(forwards)

# 打印游戏板
def print_board(board):
    print('  -1-2-3-4-5-')
    for row in range(height):
        print(row+1, '|', end='')
        for col in range(width):
            cell = board[row, col]
            if cell==0:
                print(" ", end='|')
            else:
                print(players[cell], end='|')
        if row==4:
            print('\n  -1-2-3-4-5-')
        else:
            print('\n  -----------')

# 将一个盘面用int来表达
@njit
def hash_board(board):
    h = 0
    for b in board.flat:
        h = h*3+b
    return h

# 检查游戏是否结束
@njit
def check_game_over(board, cache):
    board_hash_id =hash_board(board)
    if board_hash_id in cache:
        return cache[board_hash_id]
       
    result = False 
    if 5 in np.abs(np.sum(board, axis=0)):
        result = True
    elif 5 in np.abs(np.sum(board, axis=1)):
        result = True
    elif 5 == np.abs(np.sum(np.diag(board))):
        result = True
    elif 5 == np.abs(np.sum(np.diag(np.fliplr(board)))):
        result = True
        
    cache[board_hash_id]=result
    return result
            
# 计算游戏输赢得分
@njit
def evaluate(board, curr_player_id):
    # 检查行，优先检查对手
    sum_borad = np.sum(board, axis=0)
    if np.any(sum_borad == -curr_player_id*width): return -1

    # 检查列
    sum_borad = np.sum(board, axis=1)
    if np.any(sum_borad == -curr_player_id*height): return -1

    # 检查斜对角
    if np.all(np.diag(board) == -curr_player_id): return -1
    
    # 检查反斜对角
    if np.all(np.diag(np.fliplr(board)) == -curr_player_id): return -1
    
    return 1

# 获得可以用的坐标
@njit
def get_availables_actions(board, curr_player_id):
    availables_actions=[]
    for row in range(height):
        for col in range(width):
            if (row == 0 or col==0 or row==4 or col==4) and board[row, col] != -curr_player_id:
                availables_actions.append((row, col))
    return availables_actions
            
# 移动
@njit
def step(board, curr_player_id, row, col, forward):
    if forward==0: #w
        c = board[:,col]
        for i in range(row, 0, -1):
            c[i]=c[i-1]
        c[0] = curr_player_id
    if forward==1: #s
        c = board[:,col]
        for i in range(row, height-1):
            c[i]=c[i+1]
        c[height-1] = curr_player_id
    if forward==2: #a
        c = board[row]
        for i in range(col, 0, -1):
            c[i]=c[i-1]
        c[0] = curr_player_id
    if forward==3: #d
        c = board[row]
        for i in range(col, width-1):
            c[i]=c[i+1]
        c[width-1] = curr_player_id

# 玩家行动
def player_move(board, curr_player_id):
    availables_actions = get_availables_actions(board, curr_player_id)
    row=col=0
    while True:
        row = input('请输入行号(1-5):')
        if not row.isdigit():
            print('无效的整数，请重新输入。')
            continue
        row = int(row) -1   
                 
        col = input('请输入列号(1-5):')
        if not col.isdigit():
            print('无效的整数，请重新输入。')
            continue
        col = int(col) -1            
        
        if (row,col) not in availables_actions:            
            print('无效的位置，请重新输入。')
            continue
        else:
            break
        
    while True:
        forward = input('请输入需要移动的方向(w|a|s|d):').strip().lower()
        if forward not in forwards:
            print('无效的方向，请重新输入。')
            continue
        else:
            break
    print("玩家:", players[curr_player_id], '坐标:' ,(row+1, col+1), '方向：', forward)    
    step(board, curr_player_id, row, col, forwards.index(forward))

# 最大值最小值算法
# 返回格式[row, col, forward, value]
@njit(parallel=True)
def minimax(board, depth, curr_player_id, maximizing_player, cache):      
    if check_game_over(board, cache):
        # 是对上一个调用的判断
        v = evaluate(board, -curr_player_id)*100
        return np.array([-1, -1, -1, v], dtype=np.int32)
    if depth == 0 : 
        # 如果到最后一层，直接随机模拟
        temp_user_id = curr_player_id
        for i in range(100):
            availables_actions = get_availables_actions(board, temp_user_id)
            availables_actions_len = len(availables_actions)
            row,col = availables_actions[np.random.randint(0, availables_actions_len)]
            forward = np.random.randint(0, 4)
            step(board, temp_user_id, row, col, forward)
            if check_game_over(board, cache):
                v = evaluate(board, -curr_player_id)*100
                v = v-i if v>0 else v+i
                return np.array([-1, -1, -1, v], dtype=np.int32)
            temp_user_id = -temp_user_id
        return np.array([-1, -1, -1, 0], dtype=np.int32)
    
    availables_actions = get_availables_actions(board, curr_player_id)
    availables_actions_len = len(availables_actions)
    acts = np.zeros((4, availables_actions_len), dtype=np.int8)
    
    for i, (row, col) in enumerate(availables_actions):
        for forward in range(4):      
            new_board = board.copy()        
            step(new_board, curr_player_id, row, col, forward)
            result = minimax(new_board, depth-1, -curr_player_id, not maximizing_player, cache)
            acts[forward, i] = result[3]    
        
    if maximizing_player:
        vaule = np.max(acts)
    else:
        vaule = np.min(acts)
        
    p = np.where(acts==vaule)
    (best_row, best_col), best_forward = availables_actions[p[1][0]], p[0][0]   
   
    return np.array([best_row, best_col, best_forward, vaule], dtype=np.int32)

# 计算机行动
def computer_move(board, curr_player_id, cache):
    row, col, forward, value = minimax(board, 3, curr_player_id, True, cache)        
    print("计算机:", players[curr_player_id], '坐标:' ,(row+1, col+1), '方向:', forwards[forward], '价值:', value)   
    step(board, curr_player_id, row, col, forward)
    
# 主游戏循环
def play_game():
    print('欢迎来到 QUIXO 游戏！')
    cache = Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.boolean,
    )
    board = np.zeros((height, width), dtype=np.int8)
    curr_player_id = -1
    print_board(board)
    while True:
        # 玩家行动
        player_move(board, curr_player_id)
        if check_game_over(board, cache):
            v = evaluate(board, curr_player_id)
            if v==1:
                print('恭喜你，你赢了！')
            else:
                print('很遗憾，你输了！')
            break
        print_board(board)

        # 计算机行动
        curr_player_id = -curr_player_id
        computer_move(board, curr_player_id, cache)
        if check_game_over(board, cache):
            v = evaluate(board, curr_player_id)
            if v==1:
                print('很遗憾，你输了！')
            else:
                print('恭喜你，你赢了！')
            break        

        print_board(board)        
        curr_player_id = -curr_player_id

    print_board(board)
    print('游戏结束。')

# 运行游戏
play_game()