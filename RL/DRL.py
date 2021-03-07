# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:08:00 2021

@author: Thomas Tranchet
"""

import networkx as nx
import pulp
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from collections import deque

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Input
from tensorflow.keras.optimizers import Adam


class DRL():

    def __init__(self, net, source, target):
        self.iter = 5
        
        self.var_dict = {}
        self.net = net
        self.g = self.net.g
        self.source = source
        self.target = target

        self.var_dict = {}

        self.dict_res = defaultdict(dict)

        self.node_pose = self.net.node_pose

        self.expirience_replay = deque(maxlen=2000)
        
        
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

    def intialise_variable(self):
        for (i, j) in self.g.edges:
            self.var_dict[i, j] = 0

        self.dict_res = defaultdict(dict)

        self.node_pose = self.net.node_pose
        
        self.g = self.net.g

##########################################################################################
    
    def get_path_as_list_tuples(self,path):
        solve_var = [] 
        for i in path.split("_"):
            var = str(i).split('(')
            var = var[1].split(',')
            var[0] = int(var[0])
            var[1] = int(var[1].strip(')'))
            solve_var.append(tuple(var))
        return solve_var


    def draw_path(self,solve_var):
        for link in self.g.edges:
            if link in solve_var:
                self.g.edges[link[0],link[1]]['color'] = (1,0,0,1) 
    
        colors = nx.get_edge_attributes(self.g,'color').values()

        nx.draw(self.g, pos = self.node_pose, 
            edge_color=colors, 
            with_labels=True)
        ## Save chosen path ##
        plt.show()
        #plt.savefig("imgs/Q_table/Q_table_iter_{}".format(iter))
        plt.close()
        self.net.init_colors()
        
##########################################################################################
        
    def get_action(self,solve_var,current_node):
        state = self.get_state(solve_var)
        sub_list = self.get_possible_actions(current_node)
        
        if np.random.rand() <= 0.01: #Epsilon
            return random.choice(sub_list)[1]
        else :
            q_values = self.q_network.predict(state)
            return np.argmax(q_values[0])
          
    def get_possible_actions(self,current_node):
        sub_list = []
        for (i,j) in self.g.edges:
            if i == current_node:
                sub_list.append((i,j))
        return sub_list
    
    def get_list_possibles_nodes(self,current_node):
        sub_list = self.get_possible_actions(current_node)
        nodes_list = []
        for (i,j) in sub_list:
            nodes_list.append(j)
        return nodes_list
    
    def get_state(self, solve_var):
        state = np.zeros((self.net.nodes,self.net.nodes), np.int8)
        for (i,j) in self.g.edges:
            if (i,j) in solve_var:
                state[i,j] = 1
            elif i == self.source:
                state[i,j] = 2
            elif j == self.target:
                state[i,j] = 2
            else:
                state[i,j] = 0
        return state
    def f(self,seq): # Order preserving
        ''' Modified version of Dave Kirby solution '''
        seen = set()
        return [x for x in seq if x not in seen and not seen.add(x)]
    
    def give_final_reward(self,solve_var,neg_reward):
        cpt=0
        list_of_action_taken = []
        for state in self.expirience_replay:
            #print("cpt = {}".format(cpt))
            print(state)
            action_taken = solve_var[cpt][1]
            
            print("action taken = {}".format(action_taken))
            
            if (action_taken not in list_of_action_taken ):
                list_of_action_taken.add(action_taken)
                target = self.q_network.predict(state)
        
                target[0][action_taken] = target[0][action_taken] + 25/neg_reward
                self.q_network.fit(state, target, epochs=1, verbose=0)
                cpt = cpt + 1
        self.expirience_replay.clear()
        
    def train(self,solve_var,current_node,action_taken): 
        state = self.get_state(solve_var)
        #print(state)
        
        print(self.expirience_replay)
        
        self.expirience_replay.append(state) 
        
        target = self.q_network.predict(state)
        
        for j in self.get_list_possibles_nodes(current_node):
            target[0][j] = target[0][j] + 0.1

        target[0][action_taken] = target[0][action_taken] - 1
        self.q_network.fit(state, target, epochs=1, verbose=0)

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
        
        
    def _build_compile_model(self,solve_var=[]):
        state = self.get_state(solve_var)
        state_size = state[0] * state[1]

        model = Sequential()
        model.add(Input(shape=(25,)))
        model.add(Dense(self.net.nodes*2, activation='relu'))
        model.add(Dense(self.net.nodes*2, activation='relu'))
        model.add(Dense(self.net.nodes, activation='linear'))
        
        model.compile(loss='mse', optimizer= Adam(learning_rate=0.01))
        return model

##########################################################################################     
    def DRL_table(self,bdw):
        var_dict = self.var_dict
        summ = 0
        for iter in range (self.iter):
            current_node = self.source
            path = ""
            cpt = 0
            solve_var = []
            j = self.source
            while j!= self.target and cpt < 50:
                #prend une action parmis les choix possibles
                j = self.get_action (solve_var,current_node)
                i = current_node
                #condition
                
                #if (bdw + self.g.edges[i,j]['used']  <= self.g.edges[i,j]['capacity']):
                if ((i,j) in self.get_possible_actions(i)):
                    path += "{}_".format(str((i,j)))
                    solve_var.append((i,j))
                    var_dict[i,j] = 1
                    current_node = j
                    cpt = cpt + 1
                else:self.train(solve_var,current_node,j)
                    
            path = path[:-1]
            if (len(path) != 0) :
                solve_var = self.get_path_as_list_tuples(path)
    
                ## Draw chosen path red ##
                self.draw_path(solve_var)
    
                self.dict_res[iter]['Number of nodes'] = len(solve_var)
                
                # if(iter>self.iter/90):
                #     summ = summ + len(solve_var)
                
                print("solve_var len : {}".format(len(solve_var)))
                
                self.give_final_reward(solve_var,len(solve_var))
                
                self.dict_res[iter]['Sum of delay'] = sum([self.g.edges[i, j]['delay'] for i, j in solve_var])
                self.dict_res[iter]['Ratio Sum'] = sum([self.g.edges[i, j]['ratio'] for i, j in solve_var])
                self.dict_res[iter]['Squared Ratio Sum'] = sum([self.g.edges[i, j]['ratio']**2 for i, j in solve_var])
                self.dict_res[iter]['Score Sum'] = sum([self.g.edges[i, j]['score'] for i, j in solve_var])
                self.dict_res[iter]['Squared Score Sum'] = sum([self.g.edges[i, j]['score']**2 for i, j in solve_var])

                self.intialise_variable()
            
            
        # print("DRL : {}".format(summ/10))
                
                
    