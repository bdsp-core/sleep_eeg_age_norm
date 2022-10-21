import copy
from collections import OrderedDict
from itertools import product
import datetime
import multiprocessing
import os
import pickle
import sys
import timeit
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
import h5py
import matplotlib.pyplot as plt
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torch.distributions.multivariate_normal import MultivariateNormal
#from torch.distributions.constraint_registry import transform_to
from braindecode.util import np_to_var, var_to_np


def unique_keeporder(x):
    _, idx = np.unique(x, return_index=True)
    return x[np.sort(idx)]
    
    
class SleepSpecDataset(Dataset):
    """
    """
    def __init__(self, input_path, really_load=True):
        super(SleepSpecDataset, self).__init__()
        self.input_path = input_path
        #self.channel = channel
        
        # load into memory
        with h5py.File(self.input_path, 'r') as data_source:
            self.freq = data_source['freq'][:].astype(float)
            self.Fs = data_source['Fs'][()].astype(float)
            #self.channelnames = data_source['channelname'][:]
            
            if really_load:
                self.specs = data_source['spec'][:].astype(float)
            else:
                self.specs = data_source['spec']
            if not really_load:
                return
            self.subjects = data_source['subject'][:].astype(str)
            self.sleep_stages = data_source['sleep_stage'][:].astype(float)
            self.ages = data_source['age'][:].astype(float)
            self.sexs = data_source['sex'][:].astype(int)
            
        self.original_specs = np.array(self.specs, copy=True)
        self.original_sleep_stages = np.array(self.sleep_stages, copy=True)
        self.original_subjects = np.array(self.subjects, copy=True)
        self.original_ages = np.array(self.ages, copy=True)
        _, self.nfreq, _ = self.specs.shape
            
        self.unique_subjects = unique_keeporder(self.subjects)
        self.len = len(self.specs)
        
    def set_age_order(self, ages=None, order=None):
        if order is None:
            order = self.age_order
        else:
            self.age_order = order
        if ages is None:
            xx = self.original_ages
        else:
            xx = ages
        xx = xx/100.
        self.ages = np.vstack([xx**ii for ii in range(1,order+1)]).T
        
    def select_channel(self, channel_idx):
        self.specs = np.nanmean(self.original_specs[:,:,channel_idx], axis=2)
            
        # remove sharp peaks by detecting 2nd order difference
        goodids1 = np.where(~np.any(np.abs(np.diff(np.diff(self.specs,axis=1),axis=1))>7, axis=1))[0]
        print('%d / %d (%.2f%%) removed due to sharp peaks in spectrum'%(len(self.specs)-len(goodids1), len(self.specs), (len(self.specs)-len(goodids1))*100./len(self.specs)))
        
        # there are some N1, age<=10 epochs with extremely low power
        ids = (self.sleep_stages==3)&(self.original_ages<=10)
        if ids.sum()>0:
            this_spec = self.specs[ids][:,(self.freq>=4)&(self.freq<=8)]
            this_spec = np.power(10, this_spec/10.)
            theta_bandpower = 10*np.log10(this_spec.sum(axis=1)*(self.freq[1]-self.freq[0]))
            goodids2 = np.sort(np.r_[np.where(ids)[0][theta_bandpower>5], np.where(~ids)[0]])
            print('%d / %d (%.2f%%) removed due to extremely low power in age<=10 and N1'%(len(self.specs)-len(goodids2), len(self.specs), (len(self.specs)-len(goodids2))*100./len(self.specs)))
            
            goodids = sorted(set(goodids1)&set(goodids2))
        else:
            goodids = goodids1
        self.specs = self.specs[goodids]
        self.sleep_stages = self.original_sleep_stages[goodids]
        self.subjects = self.original_subjects[goodids]
        self.ages = self.ages[goodids]
        self.sexs = self.sexs[goodids]
        #self.set_age_order(ages=self.ages)
        self.len = len(self.ages)
                
    def set_data(self, specs=None, ages=None, sexs=None):
        if specs is not None and ages is not None and sexs is not None:
            assert len(self.specs)==len(self.ages)==len(self.sexs)
        if specs is not None:
            self.specs = specs
            self.len = len(self.specs)
        if sexs is not None:
            self.sexs = sexs
            self.len = len(self.sexs)
        if ages is not None:
            self.ages = ages
            self.len = len(self.ages)
        self.set_age_order(ages=ages)
            
    def summary(self, suffix=''):
        if len(suffix)>0:
            print('\n'+suffix)
        print('subject number %d'%len(self.unique_subjects))
        print('sample number %d'%self.len)

    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        spec = self.specs[idx]
        age = self.ages[idx]
            
        return {'spec': spec.astype('float32'),
                'age': age.astype('float32')}
                
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


def slice_dataset(dataset, ids):
    dataset2 = copy.deepcopy(dataset)
    for vn in ['specs', 'sleep_stages', 'subjects', 'ages', 'sexs', 'original_ages', 'original_specs']:#, 'seg_times'
        if hasattr(dataset2, vn):
            setattr(dataset2, vn, getattr(dataset, vn)[ids])
        
    dataset2.unique_subjects = unique_keeporder(dataset2.subjects)
    dataset2.len = len(dataset2.specs)
    return dataset2
        

class SpecModel(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(SpecModel, self).__init__()
        #self.freq = freq
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num  # spectral power at each frequency
        self.layer_num = len(self.hidden_num)
        #self.dropout = dropout
        #self.register_buffer('eye', Variable(th.from_numpy(np.eye(self.output_num).astype('float32')), requires_grad=False))
       
        # F(a) --> R^{#f}
        for i in range(len(self.hidden_num)):
            if i==0:
                nin = self.input_num
            else:
                nin = self.hidden_num[i-1]
            if i==len(self.hidden_num)-1:
                nout = self.hidden_num[i]*self.output_num
            else:
                nout = self.hidden_num[i]
            exec('self.dense_f%d = nn.Linear(nin, nout, bias=True)'%i)
        self.dense_output_weight_f = Parameter(th.Tensor(self.output_num, self.hidden_num[0],1))
        nn.init.xavier_normal_(self.dense_output_weight_f)
        self.dense_output_bias_f = Parameter(th.Tensor(1, self.output_num))
        nn.init.constant_(self.dense_output_bias_f, 0.)
        
        for i in range(len(self.hidden_num)):
            if i==0:
                nin = self.input_num
            else:
                nin = self.hidden_num[i-1]
            if i==len(self.hidden_num)-1:
                nout = self.hidden_num[i]*self.output_num
            else:
                nout = self.hidden_num[i]
            exec('self.dense_g%d = nn.Linear(nin, nout, bias=True)'%i)
        self.dense_output_weight_g = Parameter(th.Tensor(self.output_num, self.hidden_num[0],1))
        nn.init.xavier_normal_(self.dense_output_weight_g)
        self.dense_output_bias_g = Parameter(th.Tensor(1, self.output_num))
        nn.init.constant_(self.dense_output_bias_g, 0.)
        """
        
        # F(a,f)
        self.register_buffer('freq_', Variable(th.from_numpy(self.freq.reshape(1,-1).astype('float32')), requires_grad=False))
        self.f = nn.Sequential(OrderedDict([
                ('dense1', nn.Linear(self.input_num*2, self.hidden_num[0], bias=True)),
                #('bn1', nn.BatchNorm1d(self.hidden_num[0])),
                ('act1', nn.ELU()),
                #('dropout1', nn.Dropout(self.dropout)),
                #('dense2', nn.Linear(self.hidden_num[0], self.hidden_num[1], bias=False)),
                #('bn2', nn.BatchNorm1d(self.hidden_num[1])),
                #('act2', nn.Sigmoid()),
                #('dropout2', nn.Dropout(self.dropout)),
                ('dense_output', nn.Linear(self.hidden_num[0], 1, bias=True)),
        ]))
        self.g = nn.Sequential(OrderedDict([
                ('dense1', nn.Linear(self.input_num*2, self.hidden_num[0], bias=True)),
                #('bn1', nn.BatchNorm1d(self.hidden_num[0])),
                ('act1', nn.ELU()),
                #('dropout1', nn.Dropout(self.dropout)),
                #('dense2', nn.Linear(self.hidden_num[0], self.hidden_num[1], bias=False)),
                #('bn2', nn.BatchNorm1d(self.hidden_num[1])),
                #('act2', nn.Sigmoid()),
                #('dropout2', nn.Dropout(self.dropout)),
                ('dense_output', nn.Linear(self.hidden_num[0], 1, bias=True)),
        ])) 
        """

    #def weight_norms(self, ord='l2'):
    #    used = False
    #    for pn, w in self.named_parameters():
    #        if 'dense' in pn and 'weight' in pn:
    #            if ord=='l2':
    #                norm = th.sum(w**2)
    #            elif ord=='l1':
    #                norm = th.sum(th.abs(w))
    #            else:
    #                raise ValueError(ord)
    #            if not used:
    #                res = norm
    #                used = True
    #            else:
    #                res += norm
    #    return res
        
    def forward(self, spec, age):#, update_g=True):
        f = age
        for i in range(len(self.hidden_num)):
            f = eval('self.dense_f%d(f)'%i)
            if i<len(self.hidden_num)-1:
                f = F.elu(f)
        f = f.view(f.shape[0], self.output_num, -1)
        f = f.permute(1,0,2)
        f = F.elu(f)
        f = th.bmm(f, self.dense_output_weight_f)
        f = f.squeeze(dim=2).permute(1,0)
        f = f + self.dense_output_bias_f
        
        g = age
        for i in range(len(self.hidden_num)):
            g = eval('self.dense_g%d(g)'%i)
            if i<len(self.hidden_num)-1:
                g = F.elu(g)
        g = g.view(g.shape[0], self.output_num, -1)
        g = g.permute(1,0,2)
        g = F.elu(g)
        g = th.bmm(g, self.dense_output_weight_g)
        g = g.squeeze(dim=2).permute(1,0)
        g = g + self.dense_output_bias_g
        """
        if update_g:
            g = self.dense_g(age)
            #g = g.view(g.shape[0], self.output_num, -1)
            #g = g.permute(1,0,2)
            #g = F.elu(g)
            #g = th.bmm(g, self.dense_output_weight_g)
            #g = g.squeeze(dim=2).permute(1,0)
            #g = g + self.dense_output_bias_g
            g = self.dense_output_weight_g(g)
            g = g.view(g.shape[0], self.output_num, self.output_num)
            g = transform_to(MultivariateNormal.arg_constraints['scale_tril'])(g)
            gg = th.bmm(g, g.permute(0,2,1))
        else:
            g = self.eye
            gg = None
        
        loss = -MultivariateNormal(f, scale_tril=g).log_prob(spec)
        return loss, f, gg
        """
        
        """
        N = len(age)
        freq = self.freq_.expand(N, -1)
        freq = (freq.t().contiguous().view(-1,1)-10.)/10.
        freq = th.cat([freq, freq**2, freq**3, freq**4, freq**5], dim=1)
        age = age.repeat(len(self.freq), 1)
        X = th.cat([age, freq], dim=1)
        f = self.f(X)
        g = self.g(X)
        f = f.view(-1,N).t()
        g = g.view(-1,N).t()
        """
        
        # this is actually the loss
        expG = th.exp(g)
        negll = (spec-f)**2/expG + g
        return negll.sum(dim=1), f, expG  # assumes independent
        


class Experiment:
    def __init__(self, model=None,# reg=None, C=None,
            batch_size=32, max_epoch=10, lr=0.001, n_jobs=1,
            remember_best_metric='loss', verbose=False, n_gpu=0,
            save_base_path='models', random_state=None):
        self.model = model
        #self.reg = reg
        #self.C = C
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.lr = lr
        self.n_jobs = n_jobs
        self.remember_best_metric = remember_best_metric
        self.verbose = verbose
        self.n_gpu = n_gpu
        #self.model_constraint = model_constraint
        self.save_base_path = save_base_path
        self.random_state = random_state
        self.save_path = os.path.join(self.save_base_path,'current_best_model.pth')
        self.patience_reducelr = 2
        self.patience_stop = 5
    
    def fit(self, dataset, dataset_va=None):
        self.fitted_ = False
        self.optimizer = RMSprop(filter(lambda p:p.requires_grad, self.model.parameters()), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=self.patience_reducelr, verbose=True, threshold=1e-4, min_lr=1e-6)
        
        if self.n_gpu>0:
            self.model = self.model.cuda()
        th.manual_seed(self.random_state)
        if th.cuda.is_available() and self.n_gpu>0:
            th.cuda.manual_seed(self.random_state)
            th.cuda.manual_seed_all(self.random_state)
                
        if hasattr(self.model, 'init'):
            self.model.init()
        
        if self.n_gpu>1:
            #raise NotImplementedError
            self.model = nn.DataParallel(self.model, device_ids=list(range(self.n_gpu)))
        self.best_perf = np.inf
        self.best_epoch = self.max_epoch
        if self.n_jobs==-1:
            self.n_jobs = multiprocessing.cpu_count()
            
        gen_tr = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.n_jobs, pin_memory=False)
        if dataset_va is not None:
            gen_va = DataLoader(dataset_va, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.n_jobs, pin_memory=False)
            self.best_loss = np.inf
            self.best_epoch = 0
            self.min_delta = 1e-4
            
        st = timeit.default_timer()
        for epoch in range(self.max_epoch):
            self.run_one_epoch(epoch, gen_tr, 'train', use_gpu=self.n_gpu>0)
            
            if dataset_va is not None:
                loss_va, _ = self.run_one_epoch(0, gen_va, 'eval', use_gpu=self.n_gpu>0)
                
                # early stopping
                if (loss_va - self.best_loss) < -self.min_delta:
                    self.best_loss = loss_va
                    self.best_epoch = epoch
                    self.wait = 1
                    self.save()  # save the current best model
                else:
                    if self.wait >= self.patience_stop:
                        self.stopped_epoch = epoch + 1
                        print('\nTerminated Training for Early Stopping at Epoch %d'%self.stopped_epoch)
                        break
                    self.wait += 1
                print('[%d %s] val_loss: %g, current best: [epoch %d] %g' % (epoch+1, datetime.datetime.now(), loss_va, self.best_epoch+1, self.best_loss))
            else:
                self.save()
                
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                if type(self.scheduler)==ReduceLROnPlateau:
                    self.scheduler.step(loss_va)
                else:
                    self.scheduler.step()
                    
        et = timeit.default_timer()
        self.train_time = et-st

        if self.verbose:
            print('training time: %gs'%self.train_time)
            
        self.load()  # load the current best model
        self.fitted_ = True
        return self
        
    def run_one_epoch(self, epoch, gen, train_or_eval, use_gpu=False, evaluate_loss=True):
        if train_or_eval=='train':
            running_loss = 0.
            verbosity = 1000
            self.model.train()
        else:
            total_loss = 0.
            total_outputs = []
            self.model.eval()
            
        N = 0.
        for bi, batch in enumerate(gen):
            spec = Variable(batch['spec'])
            age = Variable(batch['age'])
            
            batch_size = len(spec)
            N += batch_size
            
            if use_gpu:
                spec = spec.cuda()
                age = age.cuda()
            
            if train_or_eval=='train':
                self.optimizer.zero_grad()
                
            outputs = self.model(spec, age)#, epoch>=1)

            if train_or_eval=='train':
                loss = th.mean(outputs[0])
                #if self.reg is not None and self.C is not None:
                #    loss += self.model.weight_norms(ord=self.reg)*self.C
                loss.backward()
                
                self.optimizer.step()
                
                running_loss += float(var_to_np(loss))
                #self.batch_loss_history.append(loss.data[0])
                if bi % verbosity == verbosity-1:
                    print('[%d, %d %s] loss: %g' % (epoch+1, bi+1, datetime.datetime.now(), running_loss / verbosity))
                    running_loss = 0.
            else:
                if evaluate_loss:
                    loss = th.sum(outputs[0])
                    #if self.reg is not None and self.C is not None:
                    #    loss += self.model.weight_norms(ord=self.reg)*self.C
                    total_loss += float(var_to_np(loss))

                if type(outputs)==tuple:
                    #outputs2 = [var_to_np(outputs[ii]) if outputs[ii] is not None else None for ii in range(len(outputs))]
                    outputs2 = [var_to_np(outputs[ii]) for ii in range(len(outputs))]
                else:
                    outputs2 = [var_to_np(outputs)]
                total_outputs.append(outputs2)
                
        if train_or_eval!='train':
            if N==0:
                N=1
            return total_loss/N, total_outputs
            
    def predict(self, D, output_id=0, return_only_loss=False):
        use_gpu = self.n_gpu>0
        if use_gpu:
            self.model = self.model.cuda()

        if self.n_jobs==-1:
            self.n_jobs = multiprocessing.cpu_count()
        
        gen = DataLoader(D, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.n_jobs, pin_memory=False)
                            
        loss, outputs = self.run_one_epoch(0, gen, 'eval', use_gpu=use_gpu, evaluate_loss=return_only_loss)
            
        if return_only_loss:
            return loss
        else:
            res = []
            if not hasattr(output_id, '__iter__'):
                output_id = [output_id]
            for oi in output_id:
                yp = [oo[oi] for oo in outputs]
                yp = np.concatenate(yp, axis=0)
                if yp.ndim==2 and yp.shape[1]==1:
                    yp = yp[:,0]
                res.append(yp)
            if len(res)==1:
                res = res[0]
            return res
            
    def evaluate(self, D):
        return self.predict(D, output_id=0, return_only_loss=True)

    def load(self, save_path=None, verbose=False):
        if save_path is None:
            save_path = self.save_path
        self.model = th.load(save_path)
        if type(self.model)==nn.DataParallel:
            self.model = self.model.module
        if self.n_gpu>0:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        self.fitted_ = True
        if verbose:
            print('model loaded from %s'%save_path)

    def save(self, save_path=None, verbose=False):
        if save_path is None:
            save_path = self.save_path
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        if type(self.model)==nn.DataParallel:
            th.save(self.model.module, save_path)
        else:
            th.save(self.model, save_path)
        if verbose:
            print('model saved to %s'%save_path)


    
if __name__=='__main__':
    ## config
    mode = sys.argv[1].lower()  # 'train' or 'test'
    assert mode in ['train', 'test']
    tosave = True
    n_gpu = 1
    n_jobs = 0
    
    model_type = 'row'#_cov'
    data_path = '../all_data_AHI15_Diag_betterartifact_apnea_removed.h5'
    result_path = 'step4_results_%s.pickle'%model_type
    best_hyperparam_path = 'step4_best_hyperparameters.pickle'
    
    batch_size = 64#*n_gpu
    lr = 0.001#*n_gpu
    max_epoch = 100
    channels = ['C', 'O']
    channel2idx = {'C':[0,1], 'O':[2,3]}
    stage2number = OrderedDict([('W',5), ('R',4), ('N1',3), ('N2',2), ('N3',1)])
    sexs = [1, 0]  # 1 for male, 0 for female
    random_state = 20
    
    tr_va_path = 'step4_tr_va_subjects.pickle'
    if os.path.exists(tr_va_path):
        print(f'Reading from {tr_va_path}')
        with open(tr_va_path, 'rb') as ff:
            subjects_tr, subjects_va = pickle.load(ff)
    else:
        print(f'Generating {tr_va_path}')
        np.random.seed(random_state)
        with h5py.File(data_path, 'r') as ff:
            subjects = ff['subject'][:].astype(str)
            ages = ff['age'][:]
            sexs = ff['sex'][:]
        unique_subjects, unique_ids = np.unique(subjects, return_index=True)
        ages = ages[unique_ids].astype(float)
        sexs = sexs[unique_ids].astype(int)
        
        ## get PID to MRN mapping
        df = pd.read_excel('/data/brain_age/brain_age_AllMGH/mycode-mgh/subject_files.xlsx')
        df = df.dropna(subset=['MRN', 'feature_path']).reset_index(drop=True)
        df['MRN'] = df.MRN.astype(str)
        df = df[df.MRN!='X'].reset_index(drop=True)
        df.MRN = df.MRN.str.replace('-', '').str.replace('.', '').str.replace('/', '').str.replace('_', '')
        df['StudyID'] = df.feature_path.apply(lambda x:os.path.basename(x))
        pid2mrn = {df.StudyID.iloc[i]:df.MRN.iloc[i] for i in range(len(df))}
        
        mrns = np.array([pid2mrn.get(x,x) for x in unique_subjects])
        unique_mrns = np.unique(mrns)
        np.random.shuffle(unique_mrns)
        Ntr = int(len(unique_mrns)*0.9)
        mrns_tr = unique_mrns[:Ntr]
        mrns_va = unique_mrns[Ntr:]
        subjects_tr = unique_subjects[np.in1d(mrns, mrns_tr)]
        subjects_va = unique_subjects[np.in1d(mrns, mrns_va)]
        
        ages_tr = ages[np.in1d(unique_subjects, subjects_tr)]
        ages_va = ages[np.in1d(unique_subjects, subjects_va)]
        _, pval = ks_2samp(ages_tr, ages_va)
        print(f'KS test for ages_tr and ages_va: p = {pval}')
        sexs_tr = sexs[np.in1d(unique_subjects, subjects_tr)]
        sexs_va = sexs[np.in1d(unique_subjects, subjects_va)]
        chi2, pval, dof, ex = chi2_contingency(np.array([
                    [np.sum(sexs_tr==1), np.sum(sexs_tr==0), np.sum(sexs_tr==-1)],
                    [np.sum(sexs_va==1), np.sum(sexs_va==0), np.sum(sexs_va==-1)],
                ]))
        print(f'chi2 test for sexs_tr and sexs_va: p = {pval}')
        """
        KS test for ages_tr and ages_va: p = 0.9303051314247356
        chi2 test for sexs_tr and sexs_va: p = 0.2977797323657306
        """
        
        with open(tr_va_path, 'wb') as ff:
            pickle.dump([subjects_tr, subjects_va], ff)
    
    ## generate test data
    
    # min, max, age_resolution, average_range
    age_info = np.array([[0, 4, 0.5/12/10],
                         [4, 18, 0.1],
                         [18, 91, 0.1],])
    te_ages = np.r_[np.arange(*age_info[0]),
                 np.arange(*age_info[1]),
                 np.arange(*age_info[2])]
    dummy_spec = np.zeros((len(te_ages), 196))###

    ## read all data 
    
    np.random.seed(random_state)
    
    if mode=='train':
        input_nums = [5,10]#[5]
        hidden_nums = [[50,50], [100,100], [50,50,50], [100,100,100]]#[[100,100]]
    
        dall = SleepSpecDataset(data_path, really_load=True)
        neonatal_ids = np.array(['chimes' in ss.lower() for ss in dall.subjects])
                
        best_loss = np.inf
        best_input_num = 0
        best_hidden_num = 0
        for ii, nums in enumerate(product(input_nums, hidden_nums)):
            input_num, hidden_num = nums
            print('\n[CV %d/%d]\tinput_num = %d\thidden_num = %s'%(ii+1, len(input_nums)*len(hidden_nums), input_num, hidden_num))
            
            dall.set_age_order(order=input_num)
            loss = []
            for stage, channel, sex in product(stage2number.keys(), channels, sexs):
                print(f'\n{channel} {stage} {sex}')
                model_path = f'models/spec_model_input{input_num}_hidden{hidden_num}_{sex}_{channel}_{stage}_{model_type}.pth'
                
                # neonatal Q = N1/N2/N3
                old_sleep_stages = np.array(dall.sleep_stages, copy=True)
                if stage2number[stage] in [1,2,3]:
                    dall.sleep_stages[neonatal_ids&(dall.sleep_stages==3)] = stage2number[stage]
                # always include neonatal in both sexs
                old_sexs = np.array(dall.sexs, copy=True)
                dall.sexs[neonatal_ids&(dall.sexs==-1)] = sex
                
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
                loss.append(exp.evaluate(dva))
                
                dall.sleep_stages = old_sleep_stages
                dall.sexs = old_sexs
            
            loss = np.mean(loss)
            if loss<best_loss:
                best_loss = loss
                best_input_num = input_num
                best_hidden_num = hidden_num
        
        with open(best_hyperparam_path, 'wb') as ff:
            pickle.dump([best_loss, best_input_num, best_hidden_num], ff)
        
    else:
        with open(best_hyperparam_path, 'rb') as ff:
            best_loss, best_input_num, best_hidden_num = pickle.load(ff)
        print('best_input_num = %s, best_hidden_num = %s'%(best_input_num, best_hidden_num))
            
        dte = SleepSpecDataset(data_path)#, really_load=False)
        dte.set_age_order(order=best_input_num)
        dte.set_data(specs=dummy_spec, ages=te_ages)
        result_specs = {}
        result_spec_stds = {}
        for stage, channel, sex in product(stage2number.keys(), channels, sexs):
            model_path = f'models/spec_model_input{best_input_num}_hidden{best_hidden_num}_{sex}_{channel}_{stage}_{model_type}.pth'
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
            
