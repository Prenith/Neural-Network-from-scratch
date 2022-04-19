from math import exp
import random
from math import sqrt
lamb= 0.8
lr= 0.5
mt=0.02
actual_expected_op_train=[]
actual_expected_op_test=[]
Train_err=[]
Test_err=[]
rmse1=[]
rmse2=[]


    
class Neuron:
    def __init__(self, av, no_weight, index_neuron):
        self.av = av
        self.index_neuron = index_neuron
        self.no_weight= no_weight
        self.grad_value= 0.0
        self.delta_weight= []
        self.o_error=[]
        self.h_error=[]
        self.neuron_weightslist = []
    
   
        for weight_number in range(no_weight):
            self.neuron_weightslist.append(random.random())
    
    def sigmoid_func(self,r):
        self.av = 1/(1+exp(-lamb*r))
        return self.av
    
    def  mult_weights(self, prevlayer):
        r=0.0
        for neurons in range(len(prevlayer)):
            r += prevlayer[neurons].av * prevlayer[neurons].neuron_weightslist[self.index_neuron]
        return self.sigmoid_func(r)


o_layer=[Neuron(0, 0, 0), Neuron(0, 0, 1)]
h_layer=[Neuron(0, len(o_layer), 0), Neuron(0, len(o_layer), 1), Neuron(0, len(o_layer), 2),  Neuron(0, len(o_layer), 3), Neuron(0, len(o_layer), 4)]
i_layer=[Neuron(0, len(h_layer), 0), Neuron(0, len(h_layer), 1)]


def FeedForward(inputs):

    for neurons in range(len(i_layer)):
        i_layer[neurons].av =inputs[neurons]
    for neurons in range(len(h_layer)-1):
        h_layer[neurons].mult_weights(i_layer)
    for neurons in range(len(o_layer)):
        o_layer[neurons].mult_weights(h_layer)
    return [o_layer[i].av for i in range(len(o_layer))] 

def BackPropagation(outputs):
#calcuate errors for output
    o_error=[]
    h_error = 0.0
    i_error =0.0
    
    for i in range(len(outputs)):
        o_error.append(outputs[i]-o_layer[i].av)
                

#calculating gradient value of outputlayer
    for i in range(len(o_layer)):
        for j in range(len(o_error)):
            o_layer[i].grad_value= lamb* o_layer[i].av *(1.0-o_layer[i].av)* o_error[j]
        
#calculating gradient vale of hiddenlayer
    for i in range(len(h_layer)-1):
        for j in range(len(o_layer)):
            h_error = o_layer[j].grad_value * h_layer[i].neuron_weightslist[j]
            h_layer[i].grad_value = lamb*h_layer[i].av * (1.0-h_layer[i].av)* h_error
            
#calculating gradient value of Input layer
    for i in range(len(i_layer)):
        for j in range(len(h_layer)-1):
            i_error = h_layer[j].grad_value * i_layer[i].neuron_weightslist[j]
            i_layer[i].grad_value = lamb*i_layer[i].av * (1.0-i_layer[i].av)* i_error

#calculating delta weights for hidden layer
    for i in range(len(h_layer)-1):
        for j in range(len(h_layer[i].delta_weight)):
            h_layer[i].delta_weight[j]= (lr *(o_layer[j].grad_value* h_layer[i].av))+(mt* h_layer[i].delta_weight[j])

#calculating delta weights for input layer
    for i in range(len(i_layer)):
        for j in range(len(i_layer[i].delta_weight)):
            i_layer[i].delta_weight[j]= (lr *(h_layer[j].grad_value* i_layer[i].av))+(mt* i_layer[i].delta_weight[j])

#updating weights of hidden layer
    for i in range(len(h_layer)-1):
        for j in range(len(h_layer[i].delta_weight)):
            h_layer[i].neuron_weightslist[j]= h_layer[i].delta_weight[j] + h_layer[i].neuron_weightslist[j]
            
#updating weights of input layer
    for i in range(len(i_layer)):
        for j in range(len(i_layer[i].delta_weight)):
            i_layer[i].neuron_weightslist[j]= i_layer[i].delta_weight[j] + i_layer[i].neuron_weightslist[j]

def errorcalc(in_list):
    calculated_errors=0.0
    for i in in_list:
        pdt_y1 = i[0][0]
        pdt_y2 = i[0][1]
        act_y1 = i[1][0]
        act_y2 = i[1][1]
        calculated_errors += ((pdt_y1-act_y1)**2 + (pdt_y2-act_y2)**2)/2
    rmse1=sqrt(calculated_errors/len(in_list))
    return rmse1

def training_testing():
# reading csv file
    with open('ce889_dataCollection_Training_set.csv', 'r') as csvfile:
         Training_set= csvfile.readlines()
         
    with open('ce889_dataCollection_Test_set.csv', 'r') as csvfile:
         Test_set= csvfile.readlines()
    print(Test_set)
    for i in Training_set:
        # rows= i.split(',')
        x1x2 = [float(r) for r in i.split(',')[:2]]
        y1y2 = [float(r) for r in i.split(',')[2:]]

        #print(rr)
        FeedForward(x1x2)
        BackPropagation(y1y2)
        

    for i in Training_set:
        # rows= i.split(',')
        x1x2 = [float(r) for r in i.split(',')[:2]]
        y1y2 = [float(r) for r in i.split(',')[2:]]
        actual_expected_op_train.append([FeedForward(x1x2), y1y2])
    
    print(errorcalc(actual_expected_op_train))
    
   
    for i in Test_set:
        # rows= i.split(',')
        x1x2 = [float(r) for r in i.split(',')[:2]]
        y1y2 = [float(r) for r in i.split(',')[2:]]
        actual_expected_op_test.append([FeedForward(x1x2), y1y2])
        
    print(errorcalc(actual_expected_op_test))

def epoch():
    epoch=10
    for i in range(epoch):
        rmse1= training_testing()
        print(rmse1)
epoch()
        