import networkx as nx
import pulp
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import math


class RL_part():

    def __init__(self, net, source, target):

        self.var_dict = {}
        self.net = net
        self.g = self.net.g
        self.source = source
        self.target = target

        self.var_dict = {}

        for (i, j) in self.g.edges:
            self.var_dict[i, j] = 0

        self.dict_res = defaultdict(dict)

        self.node_pose = self.net.node_pose

        # print(self.g.edges[0,0])


    def intialise_variable(self):
        for (i, j) in self.g.edges:
            self.var_dict[i, j] = 0

        self.dict_res = defaultdict(dict)

        self.node_pose = self.net.node_pose
        
        self.g = self.net.g
        

    def Monte_carlo(self, bdw):
        print(self.g.edges)
        for iter in range (5):
                current_node = self.source
                path = ""
                cpt = 0
                solve_var = []
                j = self.source
                
                while j!= self.target and cpt < 50:

                    #actions possibles
                    sub_list = []
                    for (i,j) in self.g.edges:
                        if i == current_node:
                            sub_list.append((i,j))

                    #prend une action parmis les choix possibles
                    (i,j) = random.choices(sub_list)[0]


                    #condition
                    if (bdw + self.g.edges[i,j]['used']  <= self.g.edges[i,j]['capacity']):
                        path += "{}_".format(str((i,j)))
                        solve_var.append((i,j))
                        self.var_dict[i,j] = 1
                        current_node = j
                        cpt = cpt + 1

                path = path[:-1]
                for i in path.split("_"):
                    var = str(i).split('(')
                    print(var)
                    var = var[1].split(',')
                    var[0] = int(var[0])
                    var[1] = int(var[1].strip(')'))
                    solve_var.append(tuple(var))

                for link in self.g.edges:
                    if link in solve_var:
                        self.g.edges[link[0],link[1]]['color'] = (1,0,0,1) 
                
                colors = nx.get_edge_attributes(self.g,'color').values()

                nx.draw(self.g, pos = self.node_pose, 
                    edge_color=colors, 
                    with_labels=True)

                plt.show()

                self.dict_res[iter]['Number of nodes'] = len(solve_var)
                self.dict_res[iter]['Sum of delay'] = sum([self.g.edges[i, j]['delay'] for i, j in solve_var])
                self.dict_res[iter]['Ratio Sum'] = sum([self.g.edges[i, j]['ratio'] for i, j in solve_var])
                self.dict_res[iter]['Squared Ratio Sum'] = sum([self.g.edges[i, j]['ratio']**2 for i, j in solve_var])
                self.dict_res[iter]['Score Sum'] = sum([self.g.edges[i, j]['score'] for i, j in solve_var])
                self.dict_res[iter]['Squared Score Sum'] = sum([self.g.edges[i, j]['score']**2 for i, j in solve_var])

                color = {}
                for i, j in self.g.edges:
                    color[i, j] = color[j, i] = (0,0,0,0.5)

                nx.set_edge_attributes(self.g, color, 'color') 
                plt.savefig("Monte_Carlo")  
                plt.close()



##########################################################################################
##########################################################################################
    """
     TODO : create reward table "en fonction" of the caracteristics chosen by the user, each node will have a reward compared
            to other nodes following the current state [-1;1]

            The final state will have a reward of 1.5 or 2
    """
        
    def get_action (self,epsilon,model, state, current_node):
        # print(q_table[current_node])
        values = model[current_node]
        # print(values)
        max_value = np.max(values)
        # print(q_table[state, current_node, :])
        # actions = [a for a in range(len(values))]
        # greedy_actions = [a for a in range(len(values)) if values[a] == max_value]
        # print(list(filter(lambda x : x[0]==current_node,self.g.edges)))
        actions = list(map(lambda x : x[1],filter(lambda x : x[0]==current_node,self.g.edges)))
        # print(state)
        # print(current_node)
        # print(actions)
        # print(values)
        greedy_actions = []
        for i in range(len(values)):
            if i in actions:
                greedy_actions.append(values[i])
        # greedy_actions = list(filter(lambda x: x==max_value, range(len(values))))


        # print("greedy_actions : ".format(greedy_actions))
        # Explore or get greedy
        if (random.random() < epsilon):
            return random.choice(actions)
        else:
            return random.choice(greedy_actions)
##########################################################################################
    # Get a model
    def get_model(self,env):
        return np.zeros((env.observation_space.n, env.action_space.n))

##########################################################################################
    # Exploration rate
    def get_epsilon(self,t, min_epsilon, divisor=25):
        return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / divisor)))
##########################################################################################
    # Learning rate
    def get_alpha(t, min_alpha, divisor=25):
        return max(min_alpha, min(1.0, 1.0 - math.log10((t + 1) / divisor)))

##########################################################################################
    def rewards(self):
        # print("table creation")
        print(self.net.caracteristique)
        reward_table = np.zeros((25,25),dtype=float )
        for i,j in self.g.edges:
            reward_table[i,j] = self.g.edges[i,j][self.net.caracteristique]
            # print((i,j))

    
        for i in range(len(reward_table)):
            # print('hello')
            list_ = reward_table[i]
            # print(max(list_))
            # print(min(list_))

        
            for j in range(len(reward_table)):

                value = reward_table[i][j]
                # print(value)
                # print(float(value - min(reward_table[i])) / float(max(reward_table[i]) - min(reward_table[i])))
                reward_table[i][j] = float(value - min(reward_table[i])) / float(max(reward_table[i]) - min(reward_table[i]))
                # print(reward_table[i][j])
        return reward_table
##########################################################################################
    # Update the model (Q-table)
    def update(self,model, current_state, reward, current_action, alpha=0.4, gamma=0.95): 
        rewards = self.rewards()
        reward_c = rewards[current_state][current_action]
        reward_n = max(rewards[current_state])
        print(model[current_state, current_action] + alpha * ((reward_c + gamma * reward_n) - model[current_state, current_action]))
        model[current_state, current_action] = model[current_state, current_action] + alpha * ((reward_c + gamma * reward_n) - model[current_state, current_action])
        print(model)
        return model
##########################################################################################
    def Q_table(self,bdw):
        #Initialisation of our matrice of variables
        # print(self.g.edges)
        # print(self.g.edges[0])
        var_array = np.zeros((25,25),np.float16 )
        var_dict = {}
        for (i, j) in self.g.edges:
            var_dict[i, j] = 0
        # Get a model (Q table)
        q_table = np.zeros(shape=(25,25,25,25), dtype=np.float16)
        model = q_table[self.source,self.target,:]
        # print(model)
        # print(q_table.shape)
        # print(q_table)
        dict_res = defaultdict(dict)

        for iter in range (10):
            current_node = self.source
            path = ""
            cpt = 0
            solve_var = []
            j = self.source
            
            while j!= self.target and cpt < 50:

                #actions possibles
                sub_list = []
                for (i,j) in self.g.edges:
                    if i == current_node:
                        sub_list.append((i,j))
                
                # current_node = j
                #prend une action parmis les choix possibles
                # print(model)
                j = self.get_action (1,model, self.source, current_node)
                i = current_node
                # print(j)
                # print(i)
                # print(current_node)
                # print((j))
                # print(i,j)
                #condition
                model = self.update(current_state=i, model=model, current_action=j, reward=self.rewards()[i][j])
                if (bdw + self.g.edges[i,j]['used']  <= self.g.edges[i,j]['capacity']):
                    path += "{}_".format(str((i,j)))
                    solve_var.append((i,j))
                    var_dict[i,j] = 1
                    var_array[i][j] = 1
                    current_node = j
                    cpt = cpt + 1
                    
            path = path[:-1]
            for i in path.split("_"):
                var = str(i).split('(')
                print(var)
                var = var[1].split(',')
                var[0] = int(var[0])
                var[1] = int(var[1].strip(')'))
                solve_var.append(tuple(var))

            for link in self.g.edges:
                if link in solve_var:
                    self.g.edges[link[0],link[1]]['color'] = (1,0,0,1) 
            
            colors = nx.get_edge_attributes(self.g,'color').values()

            nx.draw(self.g, pos = self.node_pose, 
                edge_color=colors, 
                with_labels=True)

            plt.show()
            plt.close()

            dict_res[iter]['Number of nodes'] = len(solve_var)
            dict_res[iter]['Sum of delay'] = sum([self.g.edges[i, j]['delay'] for i, j in solve_var])
            dict_res[iter]['Ratio Sum'] = sum([self.g.edges[i, j]['ratio'] for i, j in solve_var])
            dict_res[iter]['Squared Ratio Sum'] = sum([self.g.edges[i, j]['ratio']**2 for i, j in solve_var])
            dict_res[iter]['Score Sum'] = sum([self.g.edges[i, j]['score'] for i, j in solve_var])
            dict_res[iter]['Squared Score Sum'] = sum([self.g.edges[i, j]['score']**2 for i, j in solve_var])

            color = {}
            for i, j in self.g.edges:
                color[i, j] = color[j, i] = (0,0,0,0.5)

            nx.set_edge_attributes(self.g, color, 'color')
            print(model)
            q_table[self.source,self.target] = model
        return q_table