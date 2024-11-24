import numpy as np
import pickle

import sys
import matplotlib.pyplot as plt

import os
import numpy as np
#import horovod.tensorflow as hvd
#hvd.init()

task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
# Now use 'task_id' to select the appropriate dataset or parameters.

i = task_id

W='00'

#--------------------------------------------------------------------#
    
NO= (i-1) // 1
    
#--------------------------------------------------------------------#
if    (((NO) % 25 == 0)): 
  A='00'
elif  (((NO) % 25 == 1)): 
  A='01'
elif  (((NO) % 25 == 2)):
  A='02'
elif  (((NO) % 25 == 3)):  
  A='03'
elif  (((NO) % 25 == 4)): 
  A='04'
elif  (((NO) % 25 == 5)): 
  A='05'
elif  (((NO) % 25 == 6)): 
  A='06'
elif  (((NO) % 25 == 7)):
  A='07'
elif  (((NO) % 25 == 8)):  
  A='08'
elif  (((NO) % 25 == 9)): 
  A='09'
elif  (((NO) % 25 == 10)): 
  A='10'
elif  (((NO) % 25 == 11)): 
  A='11'
elif  (((NO) % 25 == 12)):
  A='12'
elif  (((NO) % 25 == 13)):  
  A='13'
elif  (((NO) % 25 == 14)): 
  A='14'
elif  (((NO) % 25 == 15)): 
  A='15'
elif  (((NO) % 25 == 16)): 
  A='16'
elif  (((NO) % 25 == 17)):
  A='17'
elif  (((NO) % 25 == 18)):  
  A='18'
elif  (((NO) % 25 == 19)): 
  A='19'
elif  (((NO) % 25 == 20)): 
  A='20'
elif  (((NO) % 25 == 21)): 
  A='21'
elif  (((NO) % 25 == 22)):
  A='22'
elif  (((NO) % 25 == 23)):  
  A='23'
elif  (((NO) % 25 == 24)): 
  A='24'
#--------------------------------------------------------------------#
B = '00'
'''  
#--------------------------------------------------------------------#
if    (((NO) % 5 == 0)): 
  B='00'
elif  (((NO) % 5 == 1)): 
  B='01'
elif  (((NO) % 5 == 2)):
  B='02'
elif  (((NO) % 5 == 3)):  
  B='03'
elif  (((NO) % 5 == 4)): 
  B='04'
#--------------------------------------------------------------------#
'''

from data import *
#from config import aposn,aloc,aval,bposn,bloc,bval,wposn,wloc,wval

place = placepos

astr = A#sys.argv[1]
aposn = int(astr)
aloc = format(para_a[aposn],".3f")
aval = para_a[aposn]

bstr = B#sys.argv[2]
bposn = int(bstr)
#bloc = format(para_b[bposn],".3f")
#bval = para_b[bposn]
bval = 0.5 - aval
bloc = format(bval,".3f")

wstr = W#sys.argv[3]
wposn = int(wstr)
wloc = format(para_w[wposn],".3f")
wval = para_w[wposn]

pos = 'w('+format(para_w[0],".3f")+','+format(para_w[-1],".3f")+')u('+format(para_u[0],".3f")+','+format(para_u[-1],".3f")+')'

#batch_n = int(np.round((71*1000-no_models()*1000)/(7)))
#batch_s = int(np.round(0.8*int(np.round((71*1000-no_models()*1000)/(7)))))


os.chdir('/home/shajigroup2/Anantha/Factory/')
Name = 'WorkshopIV'#'C:/Users/7n7nt/Desktop/ME/HollowKnight/ML_and_Nonlocality/Result_factory/Data'#'/home/shajigroup2/Anantha/Factory/Data'#'C:/Users/7n7nt/Desktop/ME/HollowKnight/ML_and_Nonlocality/Result_factory/Data'#'/home/shajigroup2/Anantha/Factory/Data'


import config as cf
from targets import target_distribution_gen_all
from utils_nn import np_distance, np_euclidean_distance, single_run, single_evaluation, update_results
from distrs import *
#from data import para_a,para_b,para_w,para_u,aposn,aloc,aval,bposn,bloc,bval,wposn,wloc,wval

#os.chdir('C:/Users/7n7nt/Desktop/ME/HollowKnight/ML_and_Nonlocality/Result_factory/')


if __name__ == '__main__':
    # Create directories for saving stuff
    for dir in [Name+'/Repo/saved'+place+'_'+pos+'_a'+aloc+'b'+bloc+'w'+wloc+'_results',Name+'/Code_'+place+'/Result/a'+astr+'/b'+bstr+'/w'+wstr+'/saved_a'+aloc+'b'+bloc+'w'+wloc+'_models', Name+'/Code_'+place+'/Result/a'+astr+'/b'+bstr+'/w'+wstr+'/result/saved_a'+aloc+'b'+bloc+'w'+wloc+'_results', Name+'/Code_'+place+'/Result/a'+astr+'/b'+bstr+'/w'+wstr+'/saved_a'+aloc+'b'+bloc+'w'+wloc+'_configs']:
        if not os.path.exists(dir):
            os.makedirs(dir)
    # Set up the Parameters of the Neural Network (i.e. the config object)
    cf.initialize()

    cf.pnn.change_batch_size(no_models())
    #cf.change_training_set()

    # Try picking up from where training was left off. If not possible, then don't load anything, just start fresh.
    try:
        cf.pnn = cf.load_config("most_recent_pnn")
        print("\nPicking up from where we left off!\n")
        starting_sweep_id = cf.pnn.sweep_id + 1
        lostrix = np.zeros((cf.pnn.sweepnos,len(para_u)))
    except FileNotFoundError:
        print("\nStarting fresh!\n")
        starting_sweep_id = cf.pnn.sweep_id
        ending_sweep_id = cf.pnn.endsweep
        lostrix = np.zeros((cf.pnn.sweepnos,len(para_u)))
    
    # Each sweep goes through all distributions. We use different optimizer parameters in different sweeps, and load previous models.
    for sweep_id in range(starting_sweep_id, cf.pnn.sweepnos):
        cf.pnn.sweep_id = sweep_id

        # Set parameters of this training sweep.

        ## For a few sweeps, reinitialize completely.
        
        if cf.pnn.sweep_id==0:
            
            if cf.pnn.transfer:
                print("Transfer Learning")
                for i in range(cf.pnn.target_distributions.shape[0]-1,-1,-1):
                    cf.pnn.copy_best(i)   
                cf.pnn.set_starting_points(broadness_left=0, broadness_right=0)
            else:
                print("No Transfer Learning")
                cf.pnn.set_starting_points(fresh_start=True)
            
            
        if cf.pnn.sweep_id==1: #and cf.pnn.sweep_id>=0:
            if cf.pnn.transfer:
                cf.pnn.set_starting_points(broadness_left=0, broadness_right=0)
            else:
                cf.pnn.set_starting_points(fresh_start=True)
            

        ## Then for a few sweeps, start from previous best model for that distribution.
        if cf.pnn.sweep_id> 1:
            cf.pnn.set_starting_points(broadness_left=0, broadness_right=0)

        ## After a given number of sweeps, learn from all other models.
        if cf.pnn.sweep_id >= 3:
            cf.pnn.set_starting_points(broadness_left=cf.pnn.target_ids.shape[0], broadness_right=cf.pnn.target_ids.shape[0])
        '''
        ## Change to SGD.
        if cf.pnn.sweep_id == 7:
            cf.pnn.optimizer = 'sgd'
            cf.pnn.lr = 1#0.5
            cf.pnn.decay = 0
            cf.pnn.momentum = 0.2
        '''
        ## Gradually reduce learning rate for SGD for fine-tuning.
        if cf.pnn.sweep_id > 7:
            cf.pnn.lr = cf.pnn.lr * 0.95#0.9

        ## Add more phases here if you'd like!
        ## E.g. increase batch size, change the loss function to a more fine-tuned one, or change the optimizer!

        # Run single sweep
        # Loop through parameters. Convention is to start from right, since that is the least noisy distribution if I do a noise scan.
        for i in range(cf.pnn.target_distributions.shape[0]-1,-1,-1):
            # Set up new distribution
            cf.pnn.change_p_target(i)
            print('\nIn sweep {}.\nAt round {} of {} (decreasing!), with distribution {} of param {}. Target distribution:\n{}'.format(cf.pnn.sweep_id,i,cf.pnn.target_distributions.shape[0]-1,cf.pnn.target_distr_name, cf.pnn.target_ids[i],cf.pnn.p_target))

            # Run model
            values = single_run()
            model = values[0]
            loss = values[1]
            accuracy = values[2]
            val_loss = values[3]
            val_accuracy = values[4]
            lostrix[sweep_id,i] = loss[0]

            # If we loaded weights from somewhere, then compare new distance to previous one in order to know whether new model is better than previous one.
            update_results(model,i)
        # Save config of the most recently finished sweep. We will continue from here if train_multiple_sweeps is run again.
        cf.pnn.save("sweep_"+str(cf.pnn.sweep_id)+"_pnn")
        cf.pnn.save("most_recent_pnn")

    np.savetxt("./"+Name+"/Code_"+place+"/Result/a"+astr+"/b"+bstr+"/w"+wstr+"/result/saved_a"+aloc+"b"+bloc+"w"+wloc+"_results/losstxt.txt",lostrix)
    np.save("./"+Name+"/Code_"+place+"/Result/a"+astr+"/b"+bstr+"/w"+wstr+"/result/saved_a"+aloc+"b"+bloc+"w"+wloc+"_results/loss.npy",lostrix)