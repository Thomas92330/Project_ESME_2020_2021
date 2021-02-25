from networkx_class import network
import networkx as nx
from RL import RL_part
net = network(25,4)


# net.draw_initial_graph()
net.intiate_values()
net.draw_initial_graph()
# print(net.return_g())
# for connections in net.g.edges:
    # print(net.g.edges[i,j])
    # print(connections[0])

# options = list(filter(lambda x : x[0]==1,net.g.edges))
# final_option = list(map(lambda x : x[1],filter(lambda x : x[0]==1,net.g.edges)))
# print(options)
# print(final_option)
# options = list(filter(lambda x: x[0]==1, ))
# print(net.g.edges[0,1])
# RL_MC = RL_part(net, 1,2)
# RL_MC.Monte_carlo(1)

bdw = 1
RL_Q = RL_part(net, 1,2)
RL_Q.Q_table(bdw)
RL_Q.Monte_carlo(bdw)

