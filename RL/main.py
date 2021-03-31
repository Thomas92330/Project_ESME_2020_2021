from networkx_class import network
from RL import RL_part
from DRL import DRL
from DRL_Conv2D import DRL_Conv2D
from DRL_LSTM import DRL_LSTM
from DRL_ConvLSTM2D import DRL_ConvLSTM2D
from MARL import MARL

net = network(25,4)

net.intiate_values()
#net.draw_initial_graph()


bdw = 1
# RL_Q = RL_part(net, 1,2)
# RL_Q.Q_table_test(bdw)
# RL_Q.Q_table_test(bdw)
# RL_Q.Monte_carlo(bdw)

# =============================================================================
# DQRL_Conv2d = DRL_Conv2D(net,1,24) 
# DQRL_ConvLSTM2D = DRL_ConvLSTM2D(net,1,24) 
# DQRL = DRL(net,1,24)
# 
# 
# 
# DQRL.DRL_table(bdw)
# DQRL_ConvLSTM2D.DRL_table(bdw)
# DQRL_Conv2d.DRL_table(bdw)
# =============================================================================

MARL_ = MARL(net,1,24)

MARL.DRL_table(bdw)