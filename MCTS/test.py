import sys
sys.path.append('..')
from ConnectXEnv.ConnectX_Game import ConnectX_MCTS
from MCTS.mcts import mcts

model_path = './temp_root.pkl'

# 和人类玩
'''
player 1 表示人类先手， 2 表示人类后手
round 表示游戏回合
'''
def play_with_human(mcts, player=1, round=1):
    player -= 1
    for i in range(round):
        board = mcts.game.getInitBoard()
        node = mcts.root
        step = 0
        mcts.game.display(board)
        while True:
            # 人类下
            if (step % 2) == player:
                print('Please input:', mcts.game.getValidMoves(board))
                move = int(input())

                # 搜索结点2s
                node = mcts.learn(node, numMCTSSims=250).get_children_with_move(move)
            else:
                node = mcts.learn(node, numMCTSSims=250)
                node, move = node.select_move()

            board, winner = mcts.game.getNextState(board, move)

            mcts.game.display(board)

            if winner != 0:
                player = -(player-1)
                if winner == player:
                    print('You won !')
                else:
                    print('You lose!')
                break
            step += 1

    mcts.save_model(model_path)


from ConnectXEnv.ConnectX_Game import ConnectX_Gym
from tqdm import tqdm
# 和gym代理玩
def play_with_coumpter(mcts, player=-1, agent_level=-1, round=5, display = False):
    env = ConnectX_Gym(switch=player, agent_level=agent_level)
    won_count = 0
    lose_count = 0
    for i in tqdm(range(round), desc='play'):
        node = mcts.root
        env.reset()
        while True:
            if display:
                env.render()
            # 获取电脑移动
            computer_move = env.get_robot_move()
            if computer_move != -1:
                # 机器人先手
                node = mcts.learn(node, numMCTSSims=250).get_children_with_move(computer_move)

            node = mcts.learn(node, numMCTSSims=250)
            node, move = node.select_move()
            obs_, reward, done, _ = env.step(move)

            if done:
                if display:
                    env.render()

                if reward == -1:
                    print('You lose!')
                    lose_count += 1
                elif reward == 1:
                    print('You won!')
                    won_count += 1
                elif reward == -10:
                    exit()
                break
        mcts.save_model(model_path)
    print('%d-%d-%d'%(won_count,round-(won_count+lose_count), lose_count))
    pass


#
# 定义MCTS类
my_mcts = mcts(ConnectX_MCTS(width=7, height=6, win_length=4))

# 加载模型
my_mcts.load_model(model_path)


# 和人类玩 1 先手， 2后手
# play_with_human(mcts=my_mcts, player=1, round=1)

# 和代理玩
play_with_coumpter(mcts=my_mcts, round=1, agent_level=2, display=True)


