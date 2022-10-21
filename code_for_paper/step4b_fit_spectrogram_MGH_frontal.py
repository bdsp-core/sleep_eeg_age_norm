from collections import OrderedDict
import os
import pickle
import sys
import numpy as np
from step4_fit_spectrogram import *

    
if __name__=='__main__':
    ## config
    mode = sys.argv[1].lower()  # 'train' or 'test'
    assert mode in ['train', 'test']
    tosave = True
    n_gpu = 1
    n_jobs = 0
    
    model_type = 'row'#_cov'
    data_path = '../all_data_AHI15_Diag_betterartifact_apnea_removed_MGH_frontal.h5'
    
    batch_size = 64#*n_gpu
    lr = 0.001#*n_gpu
    max_epoch = 100
    channels = ['F']
    channel2idx = {'F':[0,1]}
    stage2number = OrderedDict([('W',5), ('R',4), ('N1',3), ('N2',2), ('N3',1)])
    sexs = [1, 0]  # 1 for male, 0 for female
    random_state = 20
        
    ## generate test data
    
    # min, max, age_resolution, average_range
    age_info = np.array([[13, 18, 0.1],
                         [18, 91, 0.1],])
    te_ages = np.r_[np.arange(*age_info[0]),
                 np.arange(*age_info[1])]
    dummy_spec = np.zeros((len(te_ages), 196))###

    ## read all data 
    
    tr_va_path = 'step4_tr_va_subjects.pickle'
    if os.path.exists(tr_va_path):
        print(f'Reading from {tr_va_path}')
        with open(tr_va_path, 'rb') as ff:
            subjects_tr, subjects_va = pickle.load(ff)
            
    best_hyperparam_path = 'step4_best_hyperparameters.pickle'
    with open(best_hyperparam_path, 'rb') as ff:
        best_loss, input_num, hidden_num = pickle.load(ff)
    print('best_input_num = %s, best_hidden_num = %s'%(input_num, hidden_num))
            
    result_path = f'step4_results_{model_type}_MGH_frontal.pickle'
    np.random.seed(random_state)
    
    if mode=='train':
        dall = SleepSpecDataset(data_path, really_load=True)
        dall.set_age_order(order=input_num)
        
        aa=0
        for stage, channel, sex in product(stage2number.keys(), channels, sexs):
            print(f'\n{channel} {stage} {sex}')
            model_path = f'models/spec_model_input{input_num}_hidden{hidden_num}_{sex}_{channel}_{stage}_{model_type}.pth'
            aa+=1
            if aa==1:
                continue
            
            dtr = slice_dataset(dall, np.where(np.in1d(dall.subjects, subjects_tr)&(dall.sleep_stages==stage2number[stage])&(dall.sexs==sex))[0])
            dva = slice_dataset(dall, np.where(np.in1d(dall.subjects, subjects_va)&(dall.sleep_stages==stage2number[stage])&(dall.sexs==sex))[0])
            dtr.select_channel(channel2idx[channel])
            dva.select_channel(channel2idx[channel])
            #_, pval = ks_2samp(dtr.ages, dva.ages)
            #print('KS test for ages_tr and ages_tr: p = %f'%pval)
            
            model = SpecModel(input_num, hidden_num, len(dall.freq))
            exp = Experiment(model=model, batch_size=batch_size, max_epoch=max_epoch,
                        n_jobs=n_jobs, lr=lr,
                        n_gpu=n_gpu, verbose=True, random_state=random_state)
            
            exp.fit(dtr, dva)
            if tosave:
                exp.save(model_path)
        
    else:   
        dte = SleepSpecDataset(data_path)#, really_load=False)
        dte.set_age_order(order=input_num)
        dte.set_data(specs=dummy_spec, ages=te_ages)
        result_specs = {}
        result_spec_stds = {}
        for stage, channel, sex in product(stage2number.keys(), channels, sexs):
            model_path = f'models/spec_model_input{input_num}_hidden{hidden_num}_{sex}_{channel}_{stage}_{model_type}.pth'
            #if not os.path.exists(model_path):
            #    continue
            print(f'\n{channel} {stage} {sex}')
            exp = Experiment(batch_size=batch_size, n_jobs=n_jobs,
                        n_gpu=n_gpu, verbose=True, random_state=random_state)
            exp.load(model_path, verbose=True)
            
            ll, result_spec, result_spec_var = exp.predict(dte, output_id=[0,1,2])
            result_spec_std = np.sqrt(result_spec_var)
            
            result_specs[(channel,stage,sex)] = result_spec
            result_spec_stds[(channel,stage,sex)] = result_spec_std

        with open(result_path, 'wb') as ff:
            pickle.dump([result_specs, result_spec_stds, dte.freq, te_ages], ff)
            
        """
            print(np.percentile(result_spec.flatten(),(0,0.1,1,99,99.9,100)))
            print(np.percentile(result_spec_std.flatten(),(0,0.1,1,99,99.9,100)))
            plt.imshow(result_spec.T,aspect='auto',origin='lower',cmap='jet',vmin=-1,vmax=20,extent=(te_ages.min(),te_ages.max(),dte.freq.min(),dte.freq.max()));plt.show()
            plt.imshow(result_spec_std.T,aspect='auto',origin='lower',cmap='jet',vmin=0,vmax=10,extent=(te_ages.min(),te_ages.max(),dte.freq.min(),dte.freq.max()));plt.show()
            
            del dte
            dall2 = SleepSpecDataset(data_path)
            if stage2number[stage] in [1,2,3]: dall2.sleep_stages[(dall2.sleep_stages==3)&(dall2.ages<0.9)] = stage2number[stage]
            dall2 = slice_dataset(dall2, np.where(dall2.sleep_stages==stage2number[stage])[0])
            dall2.select_channel(channel2idx[channel])
            spec2=np.array([dall2.specs[(dall2.ages>=te_ages[max(0,ii-2)])&(dall2.ages<=te_ages[min(ii+2,len(te_ages)-1)])].astype(float).mean(axis=0) for ii in range(len(te_ages))])
            plt.figure();plt.imshow(spec2.T,aspect='auto',origin='lower',cmap='jet',vmin=-1,vmax=20,extent=(te_ages.min(),te_ages.max(),dall2.freq.min(),dall2.freq.max()));plt.figure();plt.imshow(result_spec.T,aspect='auto',origin='lower',cmap='jet',vmin=-1,vmax=20,extent=(te_ages.min(),te_ages.max(),dall2.freq.min(),dall2.freq.max()));plt.show()
        """

