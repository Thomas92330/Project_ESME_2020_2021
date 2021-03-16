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
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense,GlobalMaxPooling1D,GlobalMaxPooling2D, LeakyReLU, Reshape, Input, ConvLSTM2D ,Flatten,Conv2D,MaxPooling2D 
from tensorflow.keras.optimizers import Adam

plt.style.use('ggplot')

class DRL_LSTM():

    def __init__(self, net, source, target):
        self.iter = 10
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

        self.expirience_replay = np.zeros((50,25,25,1),np.int8)
        
        self.q_network = self._build_compile_model(test = 0)
  
    def intialise_variable(self):
        for (i, j) in self.g.edges:
            self.var_dict[i, j] = 0

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
        #plt.show()
        #plt.savefig("imgs/Q_table/Q_table_iter_{}".format(iter))
        plt.close()
        self.net.init_colors()
    
    def plot_graph(self,test):
        x = []
        y = []
        for i in range(self.iter):
            x.append(self.dict_res[i]["Number of nodes"])
            y.append(i)
            
        p = plt.plot(y, x)
        
        plt.title('Average on 10 last iter : {}'.format(test))
        plt.show(p)
        plt.close()
        
        
##########################################################################################
        
    def get_action(self,solve_var,current_node,mur = False):
        state = self.get_state(solve_var)
        
        sub_list = self.get_possible_actions(current_node)
        if mur == True:
            #print("give up")
            return random.choice(sub_list)[1]
        else:
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
        #print(nodes_list)
        return nodes_list
    
    def get_state(self, solve_var,add_dim = True):
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

        if add_dim:
            state = np.expand_dims(state, axis=0)
        state = np.expand_dims(state, axis=-1)
        
        #print(state.shape)
        return state
    
    def scale(self,X, x_min, x_max):
        nom = (X-X.min(axis=0))*(x_max-x_min)
        denom = X.max(axis=0) - X.min(axis=0)
        denom[denom==0] = 1
        return x_min + nom/denom 
    
    def get_epsilon(self):
        return (1.03 - self.iter_actuel / self.iter ) 
        
    def give_final_reward(self,solve_var,neg_reward):
        cpt=0
        
        for i in range(50) :
            #print("cpt = {}".format(cpt))
            
            state = self.expirience_replay[i]
            #print("i = {} state = {}".format(i,state))
            if not (np.all(state == 0)):
                
                state = np.expand_dims(state,axis=0)
                #print("state : {}".format(state))
                action_taken = solve_var[cpt][1]
                #print("action taken = {}".format(action_taken))
                
                target = self.q_network.predict(state)
                if len(solve_var) != 50:
                    #print("target with action taken : {} before : {}".format(action_taken,target[0][action_taken]))
                    #print("before : {}".format(target[0]))
                    target[0][action_taken] = target[0][action_taken] + 50/(neg_reward - cpt)
                    #print("A : {}".format(target[0]))
                    #print("target with action taken : {} after : {}".format(action_taken,target[0][action_taken]))
                else:
                    target[0][action_taken] = target[0][action_taken] - 100
                self.q_network.fit(state, target, epochs=1, verbose=0)
                target = self.q_network.predict(state)
                #print("After : {}".format(target[0]))
                cpt = cpt + 1
                
        self.expirience_replay = np.zeros((50,25,25,1),np.int8)
        
        
    def remember(self,solve_var):
        state = self.get_state(solve_var)
        #print("Rmember state shap : {}".format(state.shape))
        #print("Rmember experience replay shap : {}".format(self.expirience_replay[self.current_iter].shape))
        for i in range (50):
            result = np.all(state==self.expirience_replay[i])
            if not result:
                self.expirience_replay[self.current_iter] = state
                
    def train(self,solve_var,current_node,action_taken): 
        state = self.expirience_replay[self.current_iter]
        state = np.expand_dims(state,axis=0)
        target = self.q_network.predict(state)
        #print("train target shape :".format(target.shape))
        for j in self.get_list_possibles_nodes(current_node):
            target[0][j] = target[0][j] + 1

        target[0][action_taken] = target[0][action_taken] - 1
        self.q_network.fit(state, self.scale(target,-1,1), epochs=1, verbose=0)
        
    def _build_compile_model(self,test,solve_var=[]):
        state = self.get_state(solve_var,False)

        model = Sequential()
        
        if test==0:
            model.add(Conv2D(16, kernel_size=(2, 25), activation=LeakyReLU(),input_shape = (25,25,1)))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(LSTM(5,input_shape=(1,16)))
            
            
        elif test == 1:
            model.add(Conv2D(16, kernel_size=(2, 25), activation='relu',input_shape = (25,25,1)))
            model.add(GlobalMaxPooling2D())
        elif test == 2:
            model.add(Conv2D(16, kernel_size=(25, 2), activation=LeakyReLU(),input_shape = (25,25,1)))
            model.add(GlobalMaxPooling2D())
        elif test == 3:
            model.add(Conv2D(16, kernel_size=(25, 2), activation='relu',input_shape = (25,25,1)))
            model.add(GlobalMaxPooling2D())
        
            
        model.add(Dense(self.net.nodes, activation='linear'))
        print(model.summary())
        model.compile(loss='mse', optimizer= Adam(learning_rate=0.1))
        return model

##########################################################################################     
    def DRL_table(self,bdw):
        test_list = [0]
        for test in test_list:
            self.q_network = self._build_compile_model(test)
            var_dict = self.var_dict
            summ = 0
            iter = -1
            while iter < (self.iter):
                iter = iter + 1 
                self.iter_actuel = iter + 1
                current_node = self.source
                path = ""
                cpt = 0
                cpt_murs = 0
                solve_var = []
                j = self.source
                
                flag_finished = False
                while flag_finished != True and cpt < 50:
                    self.current_iter = cpt
                    if cpt_murs > 50:
                        j = self.get_action(solve_var,current_node,True)
                    else:
                        j = self.get_action (solve_var,current_node)
                    i = current_node
                    
                    #condition
                    #if (bdw + self.g.edges[i,j]['used']  <= self.g.edges[i,j]['capacity']):
                    if ((i,j) in self.get_possible_actions(i)):
                        path += "{}_".format(str((i,j)))
                        solve_var.append((i,j))
                        var_dict[i,j] = 1
                        current_node = j
                        cpt_murs = 0
                        cpt = cpt + 1
                        if j == self.target:
                            flag_finished = True
                        self.remember(solve_var)
                    else:
                        cpt_murs = cpt_murs + 1
                path = path[:-1]
                
                # if cpt == 50:
                #     print("ECHEC")
                    
                if (len(path) != 0) :
                    solve_var = self.get_path_as_list_tuples(path)
        
                    ## Draw chosen path red ##
                    self.draw_path(solve_var)
                    self.dict_res[iter]['Number of nodes'] = len(solve_var)
                    
                    if self.iter_actuel > 90:
                        summ = summ + len(solve_var)
                    # if self.iter_actuel > 98:
                    #     print(path)
                    #summ = summ + len(solve_var)
                    #print(self.get_epsilon())
                    #print("solve_var len : {}".format(len(solve_var)))
                    
                    self.give_final_reward(solve_var,len(solve_var))
                    
                    self.dict_res[iter]['Sum of delay'] = sum([self.g.edges[i, j]['delay'] for i, j in solve_var])
                    self.dict_res[iter]['Ratio Sum'] = sum([self.g.edges[i, j]['ratio'] for i, j in solve_var])
                    self.dict_res[iter]['Squared Ratio Sum'] = sum([self.g.edges[i, j]['ratio']**2 for i, j in solve_var])
                    self.dict_res[iter]['Score Sum'] = sum([self.g.edges[i, j]['score'] for i, j in solve_var])
                    self.dict_res[iter]['Squared Score Sum'] = sum([self.g.edges[i, j]['score']**2 for i, j in solve_var])
                else:
                    #print("no found path for iter : {}".format(iter))
                    iter = iter - 1
                self.intialise_variable()
            self.plot_graph(summ/10)
            #print("{} : {}".format(test,summ/10))
                
                
    