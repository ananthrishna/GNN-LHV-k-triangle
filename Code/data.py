import numpy as np
import os
import sys
'''
sys.path.append('../../../..')

from Data import *

para_a = parap_a
para_b = parap_b
para_w = parap_w
para_u = parap_u
place = placep

'''
#para = np.array([0.500,0.600,0.630,0.7,0.785,0.8,0.85,0.9,1.0])
para_a = np.linspace(0.0,0.5,25)
#para_b = np.array([0.0,0.3,0.5,0.7,1.0]) #not used
para_w = np.array([0.550]) #w
para_u = np.array([0.875]) #u

dirn = os.path.dirname(os.path.abspath(sys.argv[0]))
placepos = dirn[-7:-5]


#"./w"+wstr+"/saved_a"+aloc+"b"+bloc+"w"+wloc+"_results/
'''
batch_s = batchp_s
batch_n = batchp_n

'''
#batch_s = 800
#batch_n = 1000
