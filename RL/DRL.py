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
from tensorflow.keras.layers import Dense, Embedding, Reshape, Input, LSTM
from tensorflow.keras.optimizers import Adam


class DRL():

    def __init__(self, net, source, target):
        self.iter = 100
        self.iter_actuel = 0
        self.current_iter = 0
        
        self.var_dict = {}
        self.net = net
        self.g = self.net.g
        self.source = source
        self.target = target

        self.var_dict = {}

        self.dict_res = defaultdict(dict)

        self.node_pose = self.net.node_pose

        self.expirience_replay = np.zeros((50,25,25),np.int8)
        
        self.q_network = self._build_compile_model(test = 0)
  
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
        
        if np.random.rand() <= self.get_epsilon(): #Epsilon
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
    
    def scale(self,X, x_min, x_max):
        nom = (X-X.min(axis=0))*(x_max-x_min)
        denom = X.max(axis=0) - X.min(axis=0)
        denom[denom==0] = 1
        return x_min + nom/denom 
    def get_epsilon(self):
        return (1 - self.iter_actuel / self.iter ) 
        
    def give_final_reward(self,solve_var,neg_reward):
        cpt=0
        
        for i in range(50) :
            #print("cpt = {}".format(cpt))
            
            state = self.expirience_replay[i]
            #print("i = {} state = {}".format(i,state))
            if not (np.all(state == 0)):
                #print("state : {}".format(state))
                action_taken = solve_var[cpt][1]
                #print("action taken = {}".format(action_taken))
                
                target = self.q_network.predict(state)
                target[0][action_taken] = target[0][action_taken] + 100/neg_reward
                self.q_network.fit(state, self.scale(target,-1,1), epochs=1, verbose=0)
                cpt = cpt + 1
                
        self.expirience_replay = np.zeros((50,25,25),np.int8)
        
        
    def remember(self,solve_var):
        state = self.get_state(solve_var)
        for i in range (50):
            result = np.all(state==self.expirience_replay[i])
            if not result:
                self.expirience_replay[self.current_iter] = state
                
    def train(self,solve_var,current_node,action_taken): 
        state = self.get_state(solve_var)        
        target = self.q_network.predict(state)
        
        for j in self.get_list_possibles_nodes(current_node):
            target[0][j] = target[0][j] + 1

        target[0][action_taken] = target[0][action_taken] - 1
        self.q_network.fit(state, self.scale(target,-1,1), epochs=1, verbose=0)
        
    def _build_compile_model(self,test,solve_var=[]):
        state = self.get_state(solve_var)

        model = Sequential()
        model.add(Input(shape=(25,)))
        if test==0:
            model.add(Dense(self.net.nodes*2, activation='relu'))
            model.add(Dense(self.net.nodes*2, activation='relu'))
        
        elif test == 1:
            model.add(Dense(self.net.nodes*2, activation='relu'))
            model.add(Dense(self.net.nodes, activation='relu'))
            model.add(Dense(self.net.nodes*2, activation='relu'))
            
        elif test==2:
            model.add(Dense(self.net.nodes*2, activation='relu'))
            model.add(Dense(self.net.nodes, activation='relu'))
            model.add(Dense(self.net.nodes*2, activation='relu'))
            model.add(Dense(self.net.nodes, activation='relu'))
            model.add(Dense(self.net.nodes*2, activation='relu'))            
            
        model.add(Dense(self.net.nodes, activation='linear'))
        
        model.compile(loss='mse', optimizer= Adam(learning_rate=0.01))
        return model

##########################################################################################     
    def DRL_table(self,bdw):
        test_list = [0,1,2]
        for test in test_list:
            self.q_network = self._build_compile_model(test)
            var_dict = self.var_dict
            summ = 0
            for iter in range (self.iter):
                self.iter_actuel = iter + 1
                current_node = self.source
                path = ""
                cpt = 0
                solve_var = []
                j = self.source
                while j!= self.target and cpt < 50:
                    self.current_iter = cpt 
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
                        self.remember(solve_var)
                    else:
                        self.train(solve_var,current_node,j)
                        
                path = path[:-1]
                if (len(path) != 0) :
                    solve_var = self.get_path_as_list_tuples(path)
        
                    ## Draw chosen path red ##
                    self.draw_path(solve_var)
                    self.dict_res[iter]['Number of nodes'] = len(solve_var)
                    
                    if self.iter_actuel > 90:
                        summ = summ + len(solve_var)
                    
                    #print(self.get_epsilon())
                    #print("solve_var len : {}".format(len(solve_var)))
                    
                    self.give_final_reward(solve_var,len(solve_var))
                    
                    self.dict_res[iter]['Sum of delay'] = sum([self.g.edges[i, j]['delay'] for i, j in solve_var])
                    self.dict_res[iter]['Ratio Sum'] = sum([self.g.edges[i, j]['ratio'] for i, j in solve_var])
                    self.dict_res[iter]['Squared Ratio Sum'] = sum([self.g.edges[i, j]['ratio']**2 for i, j in solve_var])
                    self.dict_res[iter]['Score Sum'] = sum([self.g.edges[i, j]['score'] for i, j in solve_var])
                    self.dict_res[iter]['Squared Score Sum'] = sum([self.g.edges[i, j]['score']**2 for i, j in solve_var])
    
                self.intialise_variable()
            print("{} : {}".format(test,summ/10))
                
                
    