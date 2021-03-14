from networkx_class import network
import networkx as nx
from RL import RL_part
from DRL import DRL
from DRL_Conv2D import DRL_Conv2D
net = network(25,3)

import numpy as np

net.intiate_values()
net.draw_initial_graph()


bdw = 1
# RL_Q = RL_part(net, 1,2)
# RL_Q.Q_table_test(bdw)
# RL_Q.Q_table_test(bdw)
# RL_Q.Monte_carlo(bdw)
DQRL = DRL_Conv2D(net,1,24) 
DQRL.DRL_table(bdw)

