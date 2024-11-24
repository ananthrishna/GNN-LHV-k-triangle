# The distribtions that are to called into the targets pyfile
import numpy as np
from sympy import *
import sys
import os
#from data import para_a,para_b,para_w,para_u,aposn,aloc,aval,bposn,bloc,bval,wposn,wloc,wval
from train import *
#from config import aposn,aloc,aval,bposn,bloc,bval,wposn,wloc,wval

v,u,w = symbols('v u w')


# This is for Werner State Noisyness on a Maximmaly entangled state
'''
if (aval+bval) != 0:
    M = v*Matrix([[aval,0,0,aval],[0,bval,bval,0],[0,bval,bval,0],[aval,0,0,aval]])*(1/(2*(aval+bval)))+(1-v)*Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])*(1/4)
else:
    M = v*Matrix([[aval,0,0,aval],[0,bval,bval,0],[0,bval,bval,0],[aval,0,0,aval]])+(1-v)*Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])*(1/4)
'''

if (aval < 0.25):
    pval = 0.25 + aval
    qval = 0.0
    rval = 0.25 - aval
    sval = 0.0
    M = v*Matrix([[pval,0,0,qval],[0,rval,sval,0],[0,sval,rval,0],[qval,0,0,pval]])+(1-v)*Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])*(1/4)
    
    MES = 0.5*v*Matrix([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])+(1-v)*Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])*(1/4)
    
    MMS = 0.25*v*Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])+(1-v)*Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])*(1/4)

if (aval == 0.25):
    pval = 0.25 + aval
    qval = 0.0
    rval = 0.25 - aval
    sval = 0.0
    M = v*Matrix([[pval,0,0,qval],[0,rval,sval,0],[0,sval,rval,0],[qval,0,0,pval]])+(1-v)*Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])*(1/4)
    
    MES = 0.5*v*Matrix([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])+(1-v)*Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])*(1/4)
    
    MMS = 0.25*v*Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])+(1-v)*Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])*(1/4)
    
if (aval > 0.25):
    pval = 0.5
    qval = aval
    rval = 0.0
    sval = 0.0
    M = v*Matrix([[pval,0,0,qval],[0,rval,sval,0],[0,sval,rval,0],[qval,0,0,pval]])+(1-v)*Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])*(1/4)
    
    MES = 0.5*v*Matrix([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])+(1-v)*Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])*(1/4)
    
    MMS = 0.25*v*Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])+(1-v)*Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])*(1/4)

    
from sympy.physics.quantum import TensorProduct
MMM = TensorProduct(TensorProduct(MES,M),MES)


MT = zeros(64,64)

# Since we are using Triangle network we have to make certain changes due to constraints
for i in range(64):
    for j in range(64):
        si = "{0:06b}".format(i)
        sj = "{0:06b}".format(j)
        
        listsi = list(si)
        listsi[4] = list(si)[5]
        listsi[2] = list(si)[0]
        listsi[0] = list(si)[1]
        listsi[1] = list(si)[2]
        listsi[5] = list(si)[3]
        listsi[3] = list(si)[4]
        
        listsj = list(sj)
        listsj[4] = list(sj)[5]
        listsj[2] = list(sj)[0]
        listsj[0] = list(sj)[1]
        listsj[1] = list(sj)[2]
        listsj[5] = list(sj)[3]
        listsj[3] = list(sj)[4]
        
        MT[int(''.join(listsi),2),int(''.join(listsj),2)] = MMM[i,j]
        
def no_models():
    no_models = np.linalg.matrix_rank(np.array(MT.subs(v,1)).astype(np.float64))
    return no_models

# This is the Renou measurement setting when used give the Renou distribution
Ma = Matrix([[0,sqrt(1-w*w),-w,0]])
Mb = Matrix([[0,w,sqrt(1-w*w),0]])
Mc = Matrix([[u,0,0,sqrt(1-u*u)]])
Md = Matrix([[sqrt(1-u*u),0,0,-u]])
Mat = Matrix([0,sqrt(1-w*w),-w,0])
Mbt = Matrix([0,w,sqrt(1-w*w),0])
Mct = Matrix([u,0,0,sqrt(1-u*u)])
Mdt = Matrix([sqrt(1-u*u),0,0,-u])


Mabcd = [Ma,Mb,Mc,Md]
MTabcd = [Mat,Mbt,Mct,Mdt]


def distribution_gen(parameter1, parameter2, parameter3):
    """ parameter1 is usually a parameter of distribution (not always relevant). parameter2 is usually noise."""


    """ Info: If param_c >~ 0.886 or <~0.464, there is no classical 3-local model."""
    """ In terms of c**2: above 0.785 or below 0.215 no classical 3-local model."""
        
    MeasRes = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                Lside = TensorProduct(TensorProduct(Mabcd[i],Mabcd[j]),Mabcd[k])
                Rside = TensorProduct(TensorProduct(MTabcd[i],MTabcd[j]),MTabcd[k])
                Meas = Lside*MT*Rside
                #print(Meas)
            
                Meas = Meas.subs(u,parameter1)
                Meas = Meas.subs(v,parameter2)
                Meas = Meas.subs(w,parameter3)
                MeasStr = str(Meas)
                MeasStr = MeasStr.replace('Matrix([[','')
                MeasStr = MeasStr.replace(']])','')
            
            
                MeasRes.append(MeasStr)
            

    for x in range(64):
        MeasRes[x] = float(MeasRes[x])
        
    MeasArr = np.array(MeasRes)
        
    p = MeasArr

    return p

        
            
            
            
            
 
            
 

            