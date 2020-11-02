# ConnectX代理
'''
作者：熊盛春
时间：2020.10.24
以下代理强度：
rondom_agent < rule1_agent < rule2_agent
'''


'''
# 纯随机的agent
'''
def rondom_agent(obs,conf):
    from random import choice
    a = choice([i for i in range(conf['columns']) if obs['board'][i] == 0])
    return a

'''
# 基于规则的agent
思想是：
1、判断那一步我会胜利就下那步，没有就转下一条规则
2、判断对方下那步我会输就下那步，没有就转下一条规则
3、判断下那一步之后对手下我就会输，就不下那一步，然后转下一条规则
4、判断下下一步可能胜利的走法，没有就随机
'''
def rule1_agent(obs, conf):
    import copy
    from random import choice

    # 判断输赢
    def is_win(xsc_obs_, xsc_conf_, action, mark, flag=True):

        if flag:
            xsc_obs = copy.deepcopy(xsc_obs_)
            xsc_conf = copy.deepcopy(xsc_conf_)
        else:
            xsc_obs = xsc_obs_
            xsc_conf = xsc_conf_

        # ----------------------------判断输入无效---------------------------#
        # 无效的action
        if action < 0 or action >= xsc_conf['columns']:
            return -1

        # 该列已经放满了
        if xsc_obs['board'][action] != 0:
            return -1

        # ----------------------------执行-----------------------------------#
        # 转换为一个二维数组表示
        board = []
        for column in range(xsc_conf['columns']):
            board.append([])
            for row in range(xsc_conf['rows']):
                board[column].append(xsc_obs['board'][xsc_conf['columns'] * row + column])

        # 执行action
        for i in range(xsc_conf['rows'])[::-1]:
            if board[action][i] == 0:
                board[action][i] = mark
                break

        # 更新回一维数组
        for column in range(xsc_conf['columns']):
            for row in range(xsc_conf['rows']):
                xsc_obs['board'][xsc_conf['columns'] * row + column] = int(board[column][row])

        # --------------------------判断胜利-----------------------------------#
        # 判断胜利
        # 1 判断列的胜利
        for column in range(xsc_conf['columns']):
            win_count = 0
            for row in range(xsc_conf['rows'])[::-1]:
                if xsc_obs['board'][xsc_conf['columns'] * row + column] == mark:
                    win_count += 1
                else:
                    win_count = 0

                if win_count >= xsc_conf['inarow']:
                    return 2

        # 2 判断行的胜利
        for row in range(xsc_conf['rows']):
            win_count = 0
            for column in range(xsc_conf['columns'])[::-1]:
                if xsc_obs['board'][xsc_conf['columns'] * row + column] == mark:
                    win_count += 1
                else:
                    win_count = 0

                if win_count >= xsc_conf['inarow']:
                    return 2

        # 3 判断斜对角线的胜利
        if xsc_conf['inarow'] > xsc_conf['rows'] or xsc_conf['inarow'] > xsc_conf['columns']:
            # 行列数不够直接不用比较斜对角
            pass
        else:
            max_row_col = xsc_conf['rows']
            # 用将棋盘转换为方阵，空余地方用0填充
            if xsc_conf['rows'] > xsc_conf['columns']:
                # 行多补充列
                for i in range(max_row_col - xsc_conf['columns']):
                    board.append([0] * xsc_conf['rows'])
                pass
            elif xsc_conf['rows'] < xsc_conf['columns']:
                max_row_col = xsc_conf['columns']
                # 列多补充行
                for column in range(xsc_conf['columns']):
                    for i in range(max_row_col - xsc_conf['rows']):
                        board[column].append(0)
                pass

            # =========  +45度
            for row in range(xsc_conf['inarow'] - 1, max_row_col):
                win_count = 0
                for i in range(row + 1):
                    if board[i][row - i] == mark:
                        win_count += 1
                    else:
                        win_count = 0
                    if win_count >= xsc_conf['inarow']:
                        return 2
            for column in range(1, max_row_col - xsc_conf['inarow'] + 1):
                win_count = 0
                for i in range(column, max_row_col):
                    if board[i][max_row_col - i] == mark:
                        win_count += 1
                    else:
                        win_count = 0
                    if win_count >= xsc_conf['inarow']:
                        return 2
            # =========  -45度
            for row in range(xsc_conf['inarow']):
                win_count = 0
                for i in range(row, max_row_col):
                    if board[i - row][i] == mark:
                        win_count += 1
                    else:
                        win_count = 0
                    if win_count >= xsc_conf['inarow']:
                        return 2
            for column in range(1, xsc_conf['inarow']):
                win_count = 0
                for i in range(column, max_row_col):
                    if board[i][i - column] == mark:
                        win_count += 1
                    else:
                        win_count = 0
                    if win_count >= xsc_conf['inarow']:
                        return 2

            # --------------------------判断平局-----------------------------------#
            # 判断平局
            flag = True
            for i in range(xsc_conf['columns']):
                flag &= (xsc_obs['board'][i] != 0)
            if flag:
                return 1

        return 0

    # -------------------------
    # print('我到底是谁', obs['mark'])
    # import numpy as np
    # print(np.array(obs['board']).reshape(6,7))
    sum_step = 0
    for i in obs['board']:
        if i != 0:
            sum_step += 1

    # 对手的标志
    robot_mark = 2 if obs['mark'] == 1 else 1

    # 我赢？
    for i in range(conf['columns']):
        if is_win(obs, conf, i, obs['mark']) == 2:
            # print(sum_step, '走这里我会赢', i)
            return i

    # 我输？
    for i in range(conf['columns']):
        if is_win(obs, conf, i, robot_mark) == 2:
            # print(sum_step, '不走这里我会输', i)
            return i

    # 查找下一步不会输的位置
    action_list = []
    for i in range(conf['columns']):
        if obs['board'][i] == 0:
            next_obs = copy.deepcopy(obs)    # 深复制不然会改变环境
            is_win(next_obs, conf, i, next_obs['mark'], flag=False)
            if is_win(next_obs, conf, i, robot_mark) != 2:
                action_list.append(i)
            else:
                pass
                # print(sum_step,'我下这里，下一步我会输',i)

    # 查找下一步可能会赢的位置
    for i in action_list:
        next_obs = copy.deepcopy(obs)
        is_win(next_obs, conf, i, next_obs['mark'], flag=False)
        if is_win(next_obs, conf, i, next_obs['mark']) == 2:
            # print(sum_step, '我下这里，我可能会赢', i)
            return i

    if len(action_list) == 0:
        a = choice([i for i in range(conf['columns']) if obs['board'][i] == 0])
    else:
        a = choice(action_list)

    # print(sum_step,'我随机下这里了',a)
    return int(a)

'''
# kaggle大神(1200分)基于规则的agent，比我自己写的rule1_agent强
'''
def rule2_agent(obs, conf):

    def get_results(x, y, mark, multiplier):
        """ get list of points, lowest cells and "in air" cells of a board[x][y] cell considering mark """
        # set board[x][y] as mark
        board[x][y] = mark
        results = []
        # if some points in axis already found - axis blocked
        blocked = [False, False, False, False]
        # i is amount of marks required to add points
        for i in range(conf['inarow'], 2, -1):
            # points
            p = 0
            # lowest cell
            lc = 0
            # "in air" points
            ap = 0
            # axis S -> N, only if one mark required for victory
            if i == conf['inarow'] and blocked[0] is False:
                (p, lc, ap, blocked[0]) = process_results(p, lc, ap,
                                                          check_axis(mark, i, x, lambda z: z, y + inarow_m1,
                                                                     lambda z: z - 1))
            # axis SW -> NE
            if blocked[1] is False:
                (p, lc, ap, blocked[1]) = process_results(p, lc, ap,
                                                          check_axis(mark, i, x - inarow_m1, lambda z: z + 1,
                                                                     y + inarow_m1, lambda z: z - 1))
            # axis E -> W
            if blocked[2] is False:
                (p, lc, ap, blocked[2]) = process_results(p, lc, ap,
                                                          check_axis(mark, i, x + inarow_m1, lambda z: z - 1, y,
                                                                     lambda z: z))
            # axis SE -> NW
            if blocked[3] is False:
                (p, lc, ap, blocked[3]) = process_results(p, lc, ap,
                                                          check_axis(mark, i, x + inarow_m1, lambda z: z - 1,
                                                                     y + inarow_m1, lambda z: z - 1))
            results.append((p * multiplier, lc, ap))
        # restore board[x][y] original value
        board[x][y] = 0
        return results

    def check_axis(mark, inarow, x, x_fun, y, y_fun):
        """ check axis (NE -> SW etc.) for lowest cell and amounts of points and "in air" cells """
        (x, y, axis_max_range) = get_x_y_and_axis_max_range(x, x_fun, y, y_fun)
        zeros_allowed = conf['inarow'] - inarow
        # lowest_cell = y
        # lowest_cell calculation turned off
        lowest_cell = 0
        for i in range(axis_max_range):
            x_temp = x
            y_temp = y
            zeros_remained = zeros_allowed
            marks = 0
            # amount of empty cells that are "in air" (don't have board bottom or mark under them)
            in_air = 0
            for j in range(conf['inarow']):
                if board[x_temp][y_temp] != mark and board[x_temp][y_temp] != 0:
                    break
                elif board[x_temp][y_temp] == mark:
                    marks += 1
                # board[x_temp][y_temp] is 0
                else:
                    zeros_remained -= 1
                    if (y_temp + 1) < conf['rows'] and board[x_temp][y_temp + 1] == 0:
                        in_air -= 1
                #                 if y_temp > lowest_cell:
                #                     lowest_cell = y_temp
                if marks == inarow and zeros_remained == 0:
                    return (sp, lowest_cell, in_air, True)
                x_temp = x_fun(x_temp)
                y_temp = y_fun(y_temp)
                if y_temp < 0 or y_temp >= conf['rows'] or x_temp < 0 or x_temp >= conf['columns']:
                    return (0, 0, 0, False)
            x = x_fun(x)
            y = y_fun(y)
        return (0, 0, 0, False)

    def get_x_y_and_axis_max_range(x, x_fun, y, y_fun):
        """ set x and y inside board boundaries and get max range of axis """
        axis_max_range = conf['inarow']
        while y < 0 or y >= conf['rows'] or x < 0 or x >= conf['columns']:
            x = x_fun(x)
            y = y_fun(y)
            axis_max_range -= 1
        return (x, y, axis_max_range)

    def process_results(p, lc, ap, axis_check_results):
        """ process results of check_axis function, return lowest cell and sums of points and "in air" cells """
        (points, lowest_cell, in_air, blocked) = axis_check_results
        if points > 0:
            if lc < lowest_cell:
                lc = lowest_cell
            ap += in_air
            p += points
        return (p, lc, ap, blocked)

    def get_best_cell(best_cell, current_cell):
        """ get best cell by comparing factors of cells """
        for i in range(len(current_cell["factors"])):
            # index 0 = points, 1 = lowest cell, 2 = "in air" cells
            for j in range(3):
                # if value of best cell factor is smaller than value of
                # the same factor in the current cell
                # best cell = current cell and break the loop,
                # don't compare lower priority factors
                if best_cell["factors"][i][j] < current_cell["factors"][i][j]:
                    return current_cell
                # if value of best cell factor is bigger than value of
                # the same factor in the current cell
                # break loop and don't compare lower priority factors
                if best_cell["factors"][i][j] > current_cell["factors"][i][j]:
                    return best_cell
        return best_cell

    def get_factors(results):
        """ get list of factors represented by results and ordered by priority from highest to lowest """
        factors = []
        for i in range(conf['inarow'] - 2):
            if i == 1:
                # my checker in this cell means my victory two times
                factors.append(results[0][0][i] if results[0][0][i][0] > st else (0, 0, 0))
                # opponent's checker in this cell means my defeat two times
                factors.append(results[0][1][i] if results[0][1][i][0] > st else (0, 0, 0))
                # if there are results of a cell one row above current
                if len(results) > 1:
                    # opponent's checker in cell one row above current means my defeat two times
                    factors.append(results[1][1][i] if -results[1][1][i][0] > st else (0, 0, 0))
                    # my checker in cell one row above current means my victory two times
                    factors.append(results[1][0][i] if -results[1][0][i][0] > st else (0, 0, 0))
                else:
                    for j in range(2):
                        factors.append((0, 0, 0))
            else:
                for j in range(2):
                    factors.append((0, 0, 0))
                for j in range(2):
                    factors.append((0, 0, 0))
            # consider only if there is no "in air" cells
            if results[0][1][i][2] == 0:
                # placing opponent's checker in this cell means opponent's victory
                factors.append(results[0][1][i])
            else:
                factors.append((0, 0, 0))
            # placing my checker in this cell means my victory
            factors.append(results[0][0][i])
            # central column priority
            factors.append((1 if i == 1 and shift == 0 else 0, 0, 0))
            # if there are results of a cell one row above current
            if len(results) > 1:
                # opponent's checker in cell one row above current means my defeat
                factors.append(results[1][1][i])
                # my checker in cell one row above current means my victory
                factors.append(results[1][0][i])
            else:
                for j in range(2):
                    factors.append((0, 0, 0))
        # if there are results of a cell two rows above current
        if len(results) > 2:
            for i in range(conf['inarow'] - 2):
                # my checker in cell two rows above current means my victory
                factors.append(results[2][0][i])
                # opponent's checker in cell two rows above current means my defeat
                factors.append(results[2][1][i])
        else:
            for i in range(conf['inarow'] - 2):
                for j in range(2):
                    factors.append((0, 0, 0))
        return factors

    # define my mark and opponent's mark
    my_mark = obs['mark']
    opp_mark = 2 if my_mark == 1 else 1

    # define board as two dimensional array
    board = []
    for column in range(conf['columns']):
        board.append([])
        for row in range(conf['rows']):
            board[column].append(obs['board'][conf['columns'] * row + column])

    best_cell = None
    board_center = conf['columns'] // 2
    inarow_m1 = conf['inarow'] - 1

    # standard amount of points
    sp = 1
    # "seven" pattern threshold points
    st = 1

    # start searching for best_cell from board center
    x = board_center

    # shift to right or left from board center
    shift = 0

    # searching for best_cell
    while x >= 0 and x < conf['columns']:
        # find first empty cell starting from bottom of the column
        y = conf['rows'] - 1
        while y >= 0 and board[x][y] != 0:
            y -= 1
        # if column is not full
        if y >= 0:
            # results of current cell and cells above it
            results = []
            results.append((get_results(x, y, my_mark, 1), get_results(x, y, opp_mark, 1)))
            # if possible, get results of a cell one row above current
            if (y - 1) >= 0:
                results.append((get_results(x, y - 1, my_mark, -1), get_results(x, y - 1, opp_mark, -1)))
            # if possible, get results of a cell two rows above current
            if (y - 2) >= 0:
                results.append((get_results(x, y - 2, my_mark, 1), get_results(x, y - 2, opp_mark, 1)))

            # list of factors represented by results
            # ordered by priority from highest to lowest
            factors = get_factors(results)

            # if best_cell is not yet found
            if best_cell is None:
                best_cell = {
                    "column": x,
                    "factors": factors
                }
            # compare values of factors in best cell and current cell
            else:
                current_cell = {
                    "column": x,
                    "factors": factors
                }
                best_cell = get_best_cell(best_cell, current_cell)

        # shift x to right or left from board center
        if shift >= 0: shift += 1
        shift *= -1
        x = board_center + shift

    # 防止col = row = inarow
    if best_cell == None:
        best_cell = {'column': -1}

    # return index of the best cell column
    return best_cell["column"]

'''
# mcts代理
'''
def mcts_agent(obs, conf):
    pass


agents = [rondom_agent,
          rule1_agent,
          rule2_agent]