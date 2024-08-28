import os


import idrlnet.shortcut as sc
import argparse

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
sc.__file__

#Save training dataset in trainarray

parser = argparse.ArgumentParser()
parser.add_argument('-max_iter', type=int, default=50000)
parser.add_argument('-plots', type=bool, default=True)
parser.add_argument('-savepath', type=str, default="res/")

opt = parser.parse_args()
max_iters = opt.max_iter
plot = opt.plots
datafile_path = opt.savepath

trainarray = []
i=0
for filename in os.listdir("Dataset2"):
    fname = "Dataset2/" + filename
    with open(fname) as f:
        for line in f: # read rest of lines
                trainarray.append([float(x) for x in line.split()])
        i+=1

#Save testing dataset in testarray

testarray = []
i=0
for filename in os.listdir("Test"):
    fname = "Test/" + filename
    with open(fname) as f:
        for line in f: # read rest of lines
                testarray.append([float(x) for x in line.split()])
        i+=1

#Generate .csv file with train data

def generate_data(): 
    t = []
    V = []
    g = []
    g0 = []
    area = []
    g1 = []
    k=-1
    m=-1
    
    for i in trainarray:
        if(i[0] == 0):
            k=-1
        t.append(i[0])
        V.append(i[1])
        g.append(i[2])
        g0.append(k)
        g1.append(m)
        m=k
        k=i[2]
        area.append(0.05)
    data = {'t': t, 'V': V, 'g': g, 'g0': g0,'g1': g1, 'area' : area}  
    df = pd.DataFrame(data) 
    df.to_csv("train_sample.csv", index=False)

generate_data()

#Generate .csv file with test data

def generatet_data_test(): 
    t = []
    V = []
    g = []
    g0 = []
    area = []
    g1 = []
    k=-1
    m=-1
    
    for i in testarray:
        if(i[0] == 0):
            k=-1
        t.append(i[0])
        V.append(i[1])
        g.append(i[2])
        g0.append(k)
        g1.append(m)
        m=k
        k=i[2]
        area.append(0.05)
    data = {'t': t, 'V': V, 'g': g, 'g0': g0,'g1': g1, 'area' : area}  
    df = pd.DataFrame(data) 
    df.to_csv("test_sample.csv", index=False)

generatet_data_test()

#Training sampler

@sc.datanode(name="domain")
class Beam(sc.SampleDomain):
    def __init__(self):
        points = pd.read_csv("train_sample.csv")
        self.points = {
            col: points[col].to_numpy().reshape(-1, 1) for col in points.columns
        }
        self.constraints = {"g": self.points.pop("g")}
    def sampling(self, *args, **kwargs):
        return self.points, self.constraints

#Boundary condition sampler

@sc.datanode(name="boundary")
class BoundaryC(sc.SampleDomain):
    def sampling(self, *args, **kwargs): 
        t = []
        V = []
        g0 = []
        g1 = []
        k=-1
        m=-1
        area = []
        for i in trainarray:
            if(i[0] == 0):
                t.append(i[0])
                V.append(i[1])
                g0.append(k)
                g1.append(m)
                m=k
                k= i[2]
                area.append(0.05)
        constraints = {"g": 2.3}
        data = {'t': t, 'V': V, 'g0': g0,'g1': g1, 'area' : area}        
        df = pd.DataFrame(data) 
        points1 = df
        points = {
            col: points1[col].to_numpy().reshape(-1, 1) for col in points1.columns
        }
        return points, constraints


#Testing sampler

@sc.datanode(name="test_domain")
class InferBeam(sc.SampleDomain):
    def __init__(self):
        points = pd.read_csv("test_sample.csv")
        self.points = {
            col: points[col].to_numpy().reshape(-1, 1) for col in points.columns
        }
        self.constraints = {"g": self.points.pop("g")}
    def sampling(self, *args, **kwargs):
        return self.points, self.constraints

#Setting up idrlnet net node

net = sc.get_net_node(
    inputs=(
        "V",
        "t",
        "g0",
        "g1"
    ),
    outputs=("g",),
    name="net1",
    arch=sc.Arch.mlp,
)

#Setting up idrlnet solver

s = sc.Solver(
    sample_domains=(Beam(),BoundaryC()),
    netnodes=[net],
    network_dir="network_dir",
    max_iter=max_iters,schedule_config=dict(scheduler="ExponentialLR", gamma=math.pow(0.95, 0.001))
)

#Training the network

s.solve()

#Inferring on the test data

s.sample_domains = (InferBeam(),)
points = s.infer_step({"test_domain": ["V", "t", "g", "g0","g1"]})

#Create graphs
if(plot == True):
    for k in range(0, 4):
    
        xs = points["test_domain"]["t"][k*101:(k+1)*101].detach().cpu().numpy().ravel()
        y_pred = points["test_domain"]["g"][k*101:(k+1)*101].detach().cpu().numpy().ravel()
        plt.figure(k)
        
        
        x_plot = []
        y_plot = []
        xy_data = []
        
        for i in range(k*101,(k+1)*101):
        
            x_plot.append(testarray[i][0])
            y_plot.append(testarray[i][2])
            xy_data.append([testarray[i][0],testarray[i][2]])
            
        plt.plot(x_plot, y_plot)
        plt.plot(xs, y_pred, '-o', color='r',linestyle='dotted', markersize=4)
        
        mse = mean_squared_error(y_plot, y_pred)
        print("MSE for dataset " + str(k) + ": " + str(mse))
        
        fname = "res/TestExample" + str(k) + ".dat"
        np.savetxt(fname ,  xy_data, fmt=['%10.7f','%10.7f'])





