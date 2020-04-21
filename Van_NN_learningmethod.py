# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:25:16 2019

@author: Billy
"""

def Learning(self,inputdata,desired_output):
#Calculating Activations of the training examples in the analysed layer
#and layer before it
    OverallCost = self.CostCalculator(inputdata, desired_output)
    weight_adjustment_list = []
    bias_adjustment_list = []
    
    backwardlayerpositions = []
    for i in range(self.no_of_layers-1):
        backwardlayerpositions.append(self.no_of_layers - i)
    ###GOES THROUGH EACH LAYER AND SELECTS ACTIVATIONS OF CURRENT AND
#    PREVIOUS LAYER
    for layerposition in backwardlayerpositions:
        if layerposition == self.no_of_layers:
            TrainingActivOutputCurrent = self.NormalComputation(inputdata,layerposition-1)
            TrainingActivOutputBefore = self.NormalComputation(inputdata,layerposition-2)
        elif layerposition == 2:
            TrainingActivOutputCurrent = self.NormalComputation(inputdata,layerposition-1)
            TrainingActivOutputBefore = inputdata
            desired_output = []
            for idx,output in enumerate(self.NormalComputation(inputdata,layerposition-1)):
                desired_output.append(output-normalised_desired_output_changes[idx])
        else:
            TrainingActivOutputCurrent = self.NormalComputation(inputdata,layerposition-1)
            TrainingActivOutputBefore = self.NormalComputation(inputdata,layerposition-2)
            desired_output = []
        for idx,output in enumerate(self.NormalComputation(inputdata,layerposition-1)):
            desired_output.append(output-normalised_desired_output_changes[idx])
        unnormalised_weight_adjustments=[]
        unnormalised_bias_adjustments=[]
        unadded_activ_adjustments = []
    #USING CHAIN RULE CALCULATES SENSITIVITY OF EACH CURRENT NODE
#    TO EACH PREVIOUS NODE
    #THESE DON&#39;T GET AVERAGED
        for idx,activC in enumerate(TrainingActivOutputCurrent):
            listitw = []
            listita = []
            for i,activB in enumerate(TrainingActivOutputBefore):
                z = self.layer_list[layerposition-1].neuron_list[idx].weights[i] * activB - self.layer_list[layerposition-1].neuron_list[idx].bias
            
                dzdw = activB
                #to find desire activation change
                dzda = self.layer_list[layerposition-1].neuron_list[idx].weights[i]
                a = 1/(1+math.exp(-(z)))
                dadz = a*(1-a)
                dCda = 2*(activC-desired_output[idx])
                #sensitivity to weight - for weight adjustments
                dCdw = -dzdw*dadz*dCda
        
                #sensitivity to previous activations - for desired output changes
                dCda = -dzda*dadz*dCda
                listitw.append(dCdw)
                listita.append(dCda)
            unnormalised_weight_adjustments.append(listitw)
            unadded_activ_adjustments.append(listita)
            ##
            bias_adjustment = dzdw*dCda
            ##
            unnormalised_bias_adjustments.append(bias_adjustment)
        
        totalled_desired_activ_adjustments = []
        vlist = []
        for i in range(len(unnormalised_weight_adjustments)):
            for j in range(len(unnormalised_weight_adjustments[i])):
                vlist.append(unnormalised_weight_adjustments[i][j])
        
        for idx in range(len(unadded_activ_adjustments[0])):
            list_dCdw = []
            for j in range(len(unadded_activ_adjustments)):
                list_dCdw.append(unadded_activ_adjustments[j][idx])
            totalled_desired_activ_adjustments.append(sum(list_dCdw))
                #Averaged values in each of the suggested weight changes
        x = np.linalg.norm(np.array(vlist))
        y = np.linalg.norm(np.array(unnormalised_bias_adjustments))
        z = np.linalg.norm(np.array(totalled_desired_activ_adjustments))
        #
        # Normalises and scales all of the adjustments to the exercise&#39;s overall cost
        normalised_weight_adjustments = []
        normalised_bias_adjustments = []
        normalised_desired_output_changes= []

        x = (OverallCost*5) / x
        y = (OverallCost*5) / y
        z = (OverallCost*5) / z
        normalised_weight_adjustments = [number*x for number in vlist]
        normalised_bias_adjustments = [number*y for number in unnormalised_bias_adjustments]
        normalised_desired_output_changes = [number*z for number in totalled_desired_activ_adjustments]
        
        weight_adjustment_list.append(normalised_weight_adjustments)
        bias_adjustment_list.append(normalised_bias_adjustments)
        outputs = [weight_adjustment_list,bias_adjustment_list]
        return outputs