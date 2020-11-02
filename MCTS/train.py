from ConnectXEnv.ConnectX_Game import ConnectX_MCTS
from MCTS.mcts import mcts

# 定义MCTS类
my_mcts = mcts(ConnectX_MCTS(width=7,height=6,win_length=4))

# 加载模型
my_mcts.load_model('./temp_root.pkl')

# 自我对战学习
my_mcts.self_play(numEps=100, numMCTSSims=200, display=False)

# 保存
my_mcts.save_model('./temp_root.pkl')
