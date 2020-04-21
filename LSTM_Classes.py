# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 22:38:41 2019

@author: Billy
"""

import LSTM_functions as F
import numpy as np
import math
    
class LSTM_cell:
    def __init__(self):
        self.state = []
        self.output = []
        self.forget_weights = []
        self.forget_biases = []
        self.input_weights = []
        self.input_biases = []
        self.candidate_weights = []
        self.candidate_biases = []
        self.output_weights = []
        self.output_biases = []
        self.c_g1 = []
        self.c_g2 = []
        self.weight_changes = []
        
        ####Values purely for derivatives
        self.out_gate = []
        self.d_tanh = []
        
        
        
    def LSTM(self, in_data, label = None, number_state_params=1,learn_rate=20, new = False, Learning = False):
        #prev_hid and prev_state will just be one row, but in_data will have multiple rows
        
        inpt = in_data
        
        if F.ArraySizer(inpt)[0]>1:
            timesteps_pb = F.ArraySizer(inpt)[0]
            dim1 = len(inpt[0])+number_state_params
        else:
            timesteps_pb = 1
            dim1 = len(inpt)+number_state_params
        
        if new:
            #if new, no previous
            self.prev_forg = []
            self.prev_out = []
            self.prev_input = []
            self.prev_d_tanh = []
            self.prev_dcdc = []
            self.prev_prev_state = []
            self.prev_state = []
            self.prev_combined = []
            self.prev_diff = []
            self.prev_cost = []
            
            
            self.state = np.ones((timesteps_pb,number_state_params))*0.1
            self.prev_state = np.ones((timesteps_pb,number_state_params))*0
            self.output = np.ones((timesteps_pb,number_state_params))*0.5
            
            self.forget_weights = np.ones((dim1,number_state_params))*0.1
            self.forget_biases = np.ones((timesteps_pb,number_state_params))
            
            self.input_weights = np.ones((dim1,number_state_params))*0.1
            self.input_biases = np.ones((timesteps_pb,number_state_params))
            
            self.candidate_weights = np.ones((dim1,number_state_params))*0.1
            self.candidate_biases = np.ones((timesteps_pb,number_state_params))
            
            self.output_weights = np.ones((dim1,number_state_params))*0.1
            self.output_biases = np.ones((timesteps_pb,number_state_params))
        else:
            if Learning == True:
                #save previous values
                self.prev_forg = self.forg_gate
                self.prev_out = self.out_gate
                self.prev_output = self.output
                self.prev_in = self.in_gate
                self.prev_cand = self.cand_gate
                self.prev_input = self.Input_gate
                self.prev_d_tanh = self.d_tanh
                self.prev_dcdc = self.dctt2_dct1
                self.prev_prev_state = self.prev_state
                self.prev_state = self.state
                self.prev_combined = self.combined
                self.prev_diff = self.diff
                self.prev_cost = self.cost
        
        if not type(self.output) == type([]):
            self.combined = inpt+ self.output.tolist()[0:number_state_params][0]
        else:
            output = float(self.output[0:number_state_params][0])
            self.combined = inpt + [output]
            
        self.forg_gate = F.Sigmoid(F.PointwiseAddition(np.matmul(self.combined,self.forget_weights),self.forget_biases))

        self.in_gate =  F.Sigmoid(F.PointwiseAddition(np.matmul(self.combined,self.input_weights),self.input_biases))
        self.cand_gate =  F.tanh(F.PointwiseAddition(np.matmul(self.combined,self.candidate_weights),self.candidate_biases))
        self.Input_gate = F.PointwiseMultiplication(self.in_gate,self.cand_gate)
        
        self.out_gate = F.Sigmoid(F.PointwiseAddition(np.matmul(self.combined,self.output_weights),self.output_biases))
        
        self.c_g1 = F.PointwiseMultiplication(self.forg_gate ,self.state)
        self.state = F.PointwiseAddition(self.c_g1,self.Input_gate)
        
        
        self.output = F.PointwiseMultiplication(F.tanh(self.state),self.out_gate)
        
        if not (label == None):
            self.label = label
            self.diff = [self.label - self.output[0]]
            self.cost = [0.5*(self.diff[0]**2)]
        
        if Learning == True:
            self.d_tanh = [1-((F.tanh(self.state)[0])**2)]
            self.dctt2_dct1 = F.PointwiseMultiplication(self.d_tanh,self.output)
            if new == False:
                
                #CALCULATING BACKPROP DERIVATIVES AS PER https://drive.google.com/file/d/1RTLuTLWzyGUNvXstOdYlhdN-HbnGBntg/view
                prev_output1 = [self.prev_output[0]*1.01]
                prev_output2 = [self.prev_output[0]*0.99]
                
                error1 = self.dError_dPrevOutput(prev_output1,inpt,number_state_params,self.label)
                error2 = self.dError_dPrevOutput(prev_output2,inpt,number_state_params,self.label)
                
                dE_dprev_output = (error1-error2)/(prev_output1[0]-prev_output2[0])
                
                a = (dE_dprev_output - self.prev_diff[0])
                b = a*self.prev_dcdc[0] + self.forg_gate[0]
                b_out = a
                
                c_forg = b*self.prev_prev_state[0][0]*self.prev_forg[0]*(1-self.prev_forg[0])
                c_in = b*self.prev_prev_state[0][0]*self.prev_in[0]*(1-self.prev_in[0])
                c_cand = b*self.prev_prev_state[0][0]*float(F.sech(F.PointwiseAddition(np.matmul(self.prev_combined,self.candidate_weights),self.candidate_biases)[0])[0])
                c_out = b_out*self.prev_prev_state[0][0]*self.prev_out[0]*(1-self.prev_out[0])
                self.forget_biases[0] -= - c_forg
                self.input_biases[0] -= c_in
                self.candidate_biases[0] -= c_cand
                self.output_biases[0] += c_out

                for i in range(len(self.prev_combined)):
                    change_factor = self.prev_combined[i]*self.prev_cost[0]*learn_rate
                    print
                    self.wc_forg = c_forg*change_factor
                    self.wc_in = c_in*change_factor
                    self.wc_cand = c_cand*change_factor
                    self.wc_out = c_out*change_factor
                    
                    self.forget_weights[i] -= self.wc_forg
                    self.input_weights[i] -= self.wc_in
                    self.candidate_weights[i] -= self.wc_cand
                    self.output_weights[i] -= self.wc_out
                    
                    
            
    def dError_dPrevOutput(self,prev_output_variant,inputt,n_s_p,label):
        if not type(prev_output_variant) == type([]):
            combined = inputt+ prev_output_variant.tolist()[0:n_s_p][0]
        else:
            output = float(prev_output_variant[0:n_s_p][0])
            combined = inputt + [output]
        
        forg_gate = F.Sigmoid(F.PointwiseAddition(np.matmul(combined,self.forget_weights),self.forget_biases))

        
        in_gate =  F.Sigmoid(F.PointwiseAddition(np.matmul(self.combined,self.input_weights),self.input_biases))
        cand_gate =  F.tanh(F.PointwiseAddition(np.matmul(self.combined,self.candidate_weights),self.candidate_biases))
        Input_gate = F.PointwiseMultiplication(in_gate,cand_gate)
        
        out_gate = F.Sigmoid(F.PointwiseAddition(np.matmul(combined,self.output_weights),self.output_biases))
        
        c_g1 = F.PointwiseMultiplication(forg_gate ,self.prev_state)
        state = F.PointwiseAddition(c_g1,Input_gate)
        
        
        output = F.PointwiseMultiplication(F.tanh(state),out_gate)
        
        return 0.5*((label-prev_output_variant[0])**2)
        
    def CostCalculator(self,inpt,label):
        self.LSTM(inpt,number_state_params = 1, new=False)
        cost = ((self.state[0][0] - label)**2)**0.5
        
        return cost
    
    