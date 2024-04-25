from base_model.FBA_model_Abstract import *
from pyDOE2 import *
import ast
import numpy
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import time
import botorch
import gpytorch
import numpy
import torch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints.constraints import Interval
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from botorch import fit_gpytorch_model, fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.models import SingleTaskGP
from gpytorch.priors.torch_priors import GammaPrior
from botorch.optim import optimize_acqf
from matplotlib import pyplot as plt
import argparse
import os
import matplotlib.cbook
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import logging
import multiprocessing as mp
logging.getLogger('pyomo.core').setLevel(logging.CRITICAL)

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 24

def generate_random_bounds(RSet,RSet_rev,RSet_irrev_pos,RSet_irrev_neg, UB_dict_WT, LB_dict_WT, numSam = 10, data_seed = 2):
    """
    This function generate random flux bounds to create synthetic experiments
    """

    # randomize flux bounds using Latin-Hypercube (lhs) sampling
    lhdnormc_UB = lhs(len(RSet), samples=numSam,random_state=10+data_seed)  # Latin-Hypercube (lhs)
    lhdnormc_LB = lhs(len(RSet), samples=numSam,random_state=20+data_seed+1)  # Latin-Hypercube (lhs)
    UB_rand = lhdnormc_UB.copy()  # linearly scale array lhdnorm from bound (0,1) to bound(1% UB_WT,100% UB_WT)
    LB_rand = lhdnormc_LB.copy()  # linearly scale array lhdnorm from bound (0,1) to bound(1% LB_WT,100% LB_WT)

    RSet_nutrient = ['EX_glc__D_e','EX_succ_e','EX_o2_e']#,'EX_succ_e','EX_gln__L_e','EX_glu__L_e']

    Ex_set = ['EX_ac_e',	'EX_acald_e',	'EX_akg_e',	'EX_co2_e',	'EX_etoh_e',	'EX_for_e',
            'EX_fru_e',	'EX_fum_e',	'EX_glc__D_e',	'EX_gln__L_e',	'EX_glu__L_e',	'EX_h_e',	
            'EX_h2o_e',	'EX_lac__D_e',	'EX_mal__L_e',	'EX_nh4_e',	'EX_o2_e',	'EX_pi_e',
            'EX_pyr_e',	'EX_succ_e']
    for i in range(len(lhdnormc_UB[0, :])):
                
        LB_base = 10
        UB_base = 100
        if RSet[i] in RSet_rev:
            UB_rand[:, i] = np.interp(lhdnormc_UB[:, i], (0, 1), (LB_base,UB_base))
            LB_rand[:, i] = np.interp(lhdnormc_LB[:, i], (0, 1), (-UB_base,-LB_base))
        elif RSet[i] in RSet_irrev_pos:
            UB_rand[:, i] = np.interp(lhdnormc_UB[:, i], (0, 1), (LB_base,UB_base))
            LB_rand[:, i] = np.interp(lhdnormc_LB[:, i], (0, 1), (LB_dict_WT[RSet[i]]*.5, LB_dict_WT[RSet[i]]*1))

        elif RSet[i] in RSet_irrev_neg:            
            UB_rand[:, i] = np.interp(lhdnormc_UB[:, i], (0, 1), (UB_dict_WT[RSet[i]]*.5, UB_dict_WT[RSet[i]]*1))
            LB_rand[:, i] = np.interp(lhdnormc_LB[:, i], (0, 1), (-UB_base,-LB_base))

        RSet_nutrient = ['EX_glc__D_e','EX_fru_e','EX_pyr_e','EX_mal__L_e','EX_fum_e','EX_pyr_e']

        if RSet[i] in Ex_set:
            if RSet[i] in RSet_nutrient:
                LB_rand[:, i] = np.interp(lhdnormc_LB[:, i], (0, 1), (-10,-100))
            if RSet[i] == 'EX_o2_e':
                LB_rand[:, i] = np.interp(lhdnormc_LB[:, i], (0, 1), (-10,-100))

        if RSet[i] == 'ATPM':
            LB_rand[:, i] = np.interp(lhdnormc_LB[:, i], (0, 1), (LB_dict_WT[RSet[i]], LB_dict_WT[RSet[i]]))

    return UB_rand, LB_rand

def generate_Ref_model_mp(n_exp, n_rxn, sigma,UB_rand, LB_rand, theta_seed = 0, noise_seed = 0, solvername = "gurobi", tee = False, test_ind = False):
    """
    Create FOPs with the randomized contextual inputs
    """
    
    if test_ind:
        UB_rand = np.flip(UB_rand, 0)
        LB_rand = np.flip(LB_rand, 0)
    
    n_pool = n_exp
    # create_model
    RSet,RSet_rev,RSet_irrev_pos,RSet_irrev_neg, MetSet, UB_dict, LB_dict, S_dict = read_model()
    redox_rxn_tuple = get_redox_rxn_tuple(S_dict,RSet)

    # C_RSet initialization
    obj_rxns_lst = ["Biomass gen","ATP gen","CO2 gen", "Nutrient uptake","Redox gen"] 

    Growth_dict = {("Biomass gen","BIOMASS_Ecoli_core_w_GAM"):-10} # maximize biomass production 
    CO2_dict = {("CO2 gen","EX_co2_e"):0.1} # minmize CO2 production
    redox_rxn_dict = get_redox_rxn_dict(S_dict,RSet) # minimize redox production
    ATPm_dict = {('ATP gen','ATPM'):-1} # max ATP production
    RSet_nutrient = ['EX_glc__D_e','EX_fru_e','EX_pyr_e','EX_mal__L_e','EX_fum_e','EX_pyr_e']

    Nutrient_dict = {('Nutrient uptake',i):-1/len(RSet_nutrient) for i in RSet_nutrient} # min consumption

    c_RSet = list(Growth_dict.keys()) + list(ATPm_dict.keys()) + list(CO2_dict.keys())  + list(Nutrient_dict.keys()) + list(redox_rxn_dict.keys()) 
    C_RSet = {**Growth_dict,**ATPm_dict, **CO2_dict, **Nutrient_dict, **redox_rxn_dict}

    # number of objectives
    n_obj = len(obj_rxns_lst)

    # generate reference cost vector
    np.random.seed(theta_seed+1) #10
    theta_ref = np.random.uniform(low=0.1, high=10, size=n_obj).tolist()
    theta_ref = np.random.dirichlet(alpha=[5]*len(obj_rxns_lst))

    # # hold theta_ref
    theta_ref_hold = theta_ref
    # normalized factor
    c_norm = sum(theta_ref[0:n_rxn])
    for i, val in enumerate(theta_ref):
        if i in [ii for ii in range(n_rxn)]:
            theta_ref[i] = val/c_norm
            # equal weight
            # theta_ref[i] = 1/n_rxn 
        else:
            theta_ref[i] = 0

    C_i = {}
    for i, name in enumerate(obj_rxns_lst):
        C_i[name] = theta_ref[i]

    Cn = {}
    for i, name in enumerate(obj_rxns_lst):
        if i == 0:
            Cn[name] = -10
        else:
            Cn[name] = -1


    model_lst = []
    n_exp_pool = int(n_exp/n_pool)
    Rvar_dict = {r: [] for r in RSet}
    Rex_dict = {r: [] for r in obj_rxns_lst}
    l2norm = []
    for pool in range(n_pool):
        data = {None: {

            'ni': {None: n_exp_pool}, # num of exps
            
            # sets
            'r': {None: RSet}, # reaction set
            'm': {None: MetSet}, # metabolite set
            'c': {None: obj_rxns_lst}, # objective set
            'c_RSet': {None: c_RSet}, # c_RSet set

            # params
            'S': S_dict, # stoichiometrix coef matrix
            'C': C_i, # objective weights
            # 'Cn': Cn, # normalization (scaling) factors
            'C_RSet': C_RSet, # normalization (scaling) factors
            'nobj': {None:n_obj}, # number of objective
            }
        }

        instance_mFBA = mFBA.create_instance(data)

        UB_dict = {}
        LB_dict = {}
        # print(n_exp_pool)
        for i in range(n_exp_pool):
            for idx, r in enumerate(RSet):
                # print(i,idx)
                # print(pool*n_exp_pool+i+1)
                UB_dict[pool*n_exp_pool+i+1,r] = UB_rand[pool*n_exp_pool+i][idx]
                LB_dict[pool*n_exp_pool+i+1,r] = LB_rand[pool*n_exp_pool+i][idx]

        # set R bounds and initialize R

        for i in range(1,n_exp_pool+1):
            for r in RSet:
                instance_mFBA.R[i, r].setub(UB_dict[pool*n_exp_pool+i,r])
                instance_mFBA.R[i, r].setlb(LB_dict[pool*n_exp_pool+i,r])
                instance_mFBA.R[i, r] = (UB_dict[pool*n_exp_pool+i,r] + LB_dict[pool*n_exp_pool+i,r])/2

        opt_FBA = SolverFactory(solvername)
        if solvername == "gurobi":
            opt_FBA.options['NonConvex'] = 2
        results_FBA = opt_FBA.solve(instance_mFBA, tee=tee)
        # set Rref
        count = 0
        for i in range(1,n_exp_pool+1):
            for r in RSet:
                np.random.seed(count)
                instance_mFBA.Rref[i, r] = value(instance_mFBA.R[i, r])#+np.random.normal(0, sigma)
                Rvar_dict[r].append(value(instance_mFBA.R[i, r]))
                count+=1
            for r in obj_rxns_lst:
                Rex_dict[r].append(sum(value(instance_mFBA.R[i, j]) for rr,j in instance_mFBA.c_RSet if rr==r))
        l2norm.append(value(instance_mFBA.l2_norm))
        if str(results_FBA.solver.termination_condition) == 'optimal':
            model_lst.append(instance_mFBA)
        
        #     print(pool, "BIOMASS_Ecoli_core_w_GAM", value(instance_mFBA.R[1,'BIOMASS_Ecoli_core_w_GAM']))
        #     print(pool, "ATPm", value(instance_mFBA.R[1,'BIOMASS_Ecoli_core_w_GAM']))
        # print(results_FBA.solver.termination_condition)
    Rvar_dict2 = {r:np.var(Rvar_dict[r]) for r in RSet}
    Rstd_dict2 = {r:np.std(Rvar_dict[r]) for r in RSet}
    Rmin_dict = {r:np.min(Rvar_dict[r]) for r in RSet}
    Rmax_dict = {r:np.max(Rvar_dict[r]) for r in RSet}
    Rexvar_dict = {r:np.var(Rex_dict[r]) for r in obj_rxns_lst}
    Rexmean_dict = {r:np.mean(Rex_dict[r]) for r in obj_rxns_lst}
    l2norm_var = np.var(l2norm)
    l2norm_mean = np.mean(l2norm)


    # set noise seed
    np.random.seed(noise_seed)

    model_lst_updated = []
    for m in model_lst:
        if sigma != 0:
            m.sigma =1#sigma#1#1#sigma
            # m.sigma =sigma#1#1#sigma
        count = 0
        
        for i in range(1,n_exp_pool+1):
            for r in RSet:
                if Rstd_dict2[r] >1:
                    m.Rvar[i,r] = Rvar_dict2[r]
                    m.Rref[i, r] = value(m.Rref[i, r]) + Rstd_dict2[r]*np.random.normal(0, sigma)
                    count +=1 

                else:
                    m.Rw[r] = 0
        model_lst_updated.append(m)     
    model_lst = model_lst_updated
    theta_true = theta_ref[0:n_rxn]
    return model_lst, theta_true

def simulator(model, c_vector, solvername = "gurobi", tee = False, mp_ind = False):
    """
    Create FOPs with the randomized contextual inputs
    """
    
    # update C
    for i, name in enumerate(list(model.c)):
        try:
            model.C[name] = c_vector[i]
        except:
            model.C[name] = 0

    opt_FBA = SolverFactory(solvername)
    if solvername == "gurobi":
        opt_FBA.options['NonConvex'] = 2
    opt_ind  = False
    if mp_ind:
        opt_FBA.options['Threads'] = 4
        opt_FBA.options['NodefileStart '] = 0.5
    try:
        results_FBA = opt_FBA.solve(model, tee=tee)
        if str(results_FBA.solver.termination_condition) == 'optimal':
            loss = value(model.loss)
            opt_ind = True
        else:
            loss = 1000
    except:
        loss = 1000
    return loss, opt_ind

def f_max(m, x, solvername = 'gurobi', mp_ind = False, n_p = 8): 
    """
    Function to compute loss
    """
    # ind is an indicator for printing out the timing result of the simulator
    x.append(1.0-sum(x))
    if mp_ind:
        for i, model in enumerate(m):
            m[i] = model.clone()
        p = mp.Pool(n_p)
        results = p.starmap_async(simulator, [(model, x, solvername, False, mp_ind) for model in m])
        p.close()
        p.join()
        loss = [r[0] for r in results.get()]
        opt_ind = [r[1] for r in results.get()]
        if all(opt_ind):
            opt_ind = True
        else:
            opt_ind = False
    else:
        loss = []
        opt_ind = []
        for model in m:
            loss_tmp,opt_ind_tmp = simulator(model, x, solvername = solvername, mp_ind = False)
            loss.append(loss_tmp)
            opt_ind.append(opt_ind_tmp)
        if all(opt_ind):
            opt_ind = True
        else:
            opt_ind = False
    
    loss = np.mean(loss)
    return np.array(-loss), opt_ind # we include the negative so that our goal is to max f(x), which is standard BO convention

def train_model(X, Y, nu=1.5, noiseless_obs=True):
    """
    Generate GP surrogates with dataset of X (input),Y (output)
    """
    # make sure training data has the right dimension
    if Y.ndim == 1:
        Y = Y.unsqueeze(-1)
    # outcome transform
    standardize = Standardize(m=Y.shape[-1], batch_shape=Y.shape[:-2])
    outcome_transform = standardize
    # covariance module
    covar_module = ScaleKernel(MaternKernel(nu=nu, ard_num_dims=X.shape[-1],lengthscale_prior=GammaPrior(3.0, 6.0),))
    # create likelihood
    if noiseless_obs: # noise from funciton evaluation (loss calculation with solved FOPs). Default is noiseless
        _, aug_batch_shape = SingleTaskGP.get_batch_dimensions(
            train_X=X,
            train_Y=Y,
        )
        likelihood = GaussianLikelihood(
            batch_shape=aug_batch_shape,
            noise_constraint=Interval(lower_bound=1e-5, upper_bound=1e-4),
        )
    else:
        likelihood = None
    # define the model
    model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        covar_module=covar_module,
        input_transform=Normalize(X.shape[-1]),
        likelihood=likelihood,
        outcome_transform=outcome_transform,
    )
    # call the training procedure
    model.outcome_transform.eval()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    # fit_gpytorch_model(mll)
    fit_gpytorch_mll(mll)
    # return the model
    return model

# need ability to optimize aquisition function
def optimize_one_step_acq(model, train_X, train_Y, nx, xL, xU, seed_id):
    """
    Optimize the acquisition function to query candidate solutions
    """
    # specify inequality constraints of the form x(1) + ... + x(nx) <= 1 <-> (-1)*x(1) + ... + (-1)*x(nx) >= -1
    indices = torch.tensor([i for i in range(nx)])
    coefficients = torch.tensor([-1.0]*nx, dtype=torch.float64)
    rhs = -0.99#1.0
    inequality_constraints = [(indices, coefficients, rhs)]
    
    # numpy.random.seed(seed=10)
    botorch.utils.sampling.manual_seed(seed=seed_id)
    torch.manual_seed(seed_id)
    # define current acquisition function
    best_value = train_Y.max()

    optimize_acqf_kwargs = {
                "q": 1, #batch size
                "num_restarts": 10, #10
                "raw_samples": 512, #512
                "options": {"seed": seed_id}
            }
    acqf = UpperConfidenceBound(model, beta=4.0)
    # create hard bounds
    bounds = torch.tensor([(xL[j], xU[j]) for j in range(nx)]).T
    # find the next best theta
    if nx > 1:
        numpy.random.seed(seed=seed_id)
        new_point_analytic, acq_value = optimize_acqf(acq_function=acqf, bounds=bounds, inequality_constraints=inequality_constraints, **optimize_acqf_kwargs)
    else:
        numpy.random.seed(seed=seed_id)
        new_point_analytic, acq_value = optimize_acqf(acq_function=acqf, bounds=bounds, **optimize_acqf_kwargs)
    x_next = new_point_analytic[0,:]
    return x_next, acq_value

# run optimization routine and get next evaluation
def optimize_acq_and_get_observation(model, train_X, train_Y, nx, xL, xU, mFBA, solvername, seed_id, mp_ind = False, n_p = 2):
    """
    Wrapper to optimize acquisition funciton and perform loss calculation at the new query points
    """
    st_BO_tmp = time.time()
    # run optimization to get next candidate design point
    x_next, acq_value = optimize_one_step_acq(model, train_X, train_Y, nx, xL, xU, seed_id)
    end_BO_tmp = time.time()
    
    # evaluate the true function at the next design
    st_FOPs_tmp = time.time()
    y_next, opt_ind = f_max(mFBA,x_next.numpy().tolist(), solvername = solvername, mp_ind = mp_ind, n_p = n_p)
    end_FOPs_tmp = time.time()
    
    return x_next, y_next, acq_value, opt_ind,(end_FOPs_tmp-st_FOPs_tmp),(end_BO_tmp-st_BO_tmp) 

def confidence_interval(y,N_trial):
    """
    Function to calculate confidence interval for plotting purpose
    """
    return 1.96 * y.std(axis=0) / numpy.sqrt(N_trial)


def BO_loop(mFBA, mFBA_test, n_exp, n_test,n_rxn,sigma, solvername, theta_true, neg_loss_true, nx, xL, xU, N_init = 5,N_iter = 45,N_trial = 5,nu_val = 1.5, mp_ind = False, data_seed = 2, n_p = 2):#, PL = False):
    """
    N_init: number of initial samples
    N_iter: number of iterations
    N_trial: number of BO trials

    Hyperparameters
    nu_val: hyperparamter for the Matern kernel. Default is 1.5
    """
    Base_path = 'BO_results/nrxn_%s/sigma=%s_N_trial=' %(n_rxn,sigma) + str(N_trial) + '_N_init=' + str(N_init) + '_N_iter=' + str(N_iter) + '_n_exp=' + str(n_exp)+ '_n_rxn=' + str(n_rxn) + '_data_seed=' +str(data_seed)
    try:
        os.mkdir(Base_path)
    except Exception as e:
        # print("why it doesn't print out",e)
        pass
    # iterations that store the results for the subsequent PL analysis
    PL_iter_lst = [10,25,50,100,150,200,250]

    ### MAIN LOOP
    # initialize a list for best observed value
    best_observed_all = []
    best_observed_all_test = []
    theta_est_all = []
    theta_l2_norm_lst_all = []


    

    # create list to store computation time
    comp_time_lst = []
    BO_time_lst = []
    FOPs_time_lst = []
    # loop over number of trials
    for trial in range(N_trial):

        # record computation time
        st = time.time()
        FOPs_time = 0
        BO_time = 0
        # fix random seed
        numpy.random.seed(seed=trial)
        botorch.utils.sampling.manual_seed(seed=trial)

        # create empty arrays
        nx = xL.shape[0]
        x_history = numpy.empty((0,nx))
        y_history = numpy.empty((0,1))
        y_test_history = numpy.empty((0,1))

        # create empty list for best observed value
        best_observed = []
        best_observed_test = []

        # create empty list for theta_est
        theta_est = []
        theta_l2_norm_lst = []
        # loop over initial random samples
        init_count = 0
        while init_count < N_init:
            # for i in range(N_init):
            x_next = numpy.random.dirichlet(alpha=[1]*n_rxn)
            x_next = x_next[0:-1]
            st_FOPs_tmp = time.time()
            y_next, opt_ind = f_max(mFBA,x_next.tolist(), solvername = solvername, mp_ind = mp_ind, n_p = n_p)
            end_FOPs_tmp = time.time()
            FOPs_time = FOPs_time + (end_FOPs_tmp-st_FOPs_tmp)
            if opt_ind:
                init_count += 1
                st_FOPs_tmp = time.time()
                y_test_next, _ = f_max(mFBA_test, x_next.tolist(), solvername = solvername, mp_ind = mp_ind, n_p = n_p)
                end_FOPs_tmp = time.time()
                FOPs_time = FOPs_time + (end_FOPs_tmp-st_FOPs_tmp)
                x_history = numpy.append(x_history, x_next.reshape((1,-1)), axis=0)
                y_history = numpy.append(y_history, y_next.reshape((1,-1)), axis=0)
                y_test_history = numpy.append(y_test_history, y_test_next.reshape((1,-1)), axis=0)
                theta_est.append(x_next.tolist()+ [1-sum(x_next.tolist())])
                if numpy.max(y_history).item() == y_next.reshape((1,-1)):
                    theta_est_best = x_next.tolist() + [1-sum(x_next.tolist())]
                    best_observed_tmp = y_next.item()
                    best_observed_test_tmp = y_test_next.item()
                best_observed.append(best_observed_tmp) # be sure to update best value
                best_observed_test.append(best_observed_test_tmp) # be sure to update best value
                theta_l2_norm_lst.append(np.linalg.norm(np.subtract(theta_est_best, theta_true))/(n_rxn-1)**0.5)


        # run main BO loop
        for i in range(N_iter):
            # convert data history to tensor
            train_X = torch.tensor(x_history)
            train_Y = torch.tensor(y_history)
            
            # count BO time
            st_BO_tmp = time.time()

            # call simple training function for building GP surrogate
            try:
                model = train_model(train_X, train_Y, nu=nu_val)
            except:
                print("Model training failed, so retained previous model")

            end_BO_tmp = time.time()
            BO_time = BO_time + (end_BO_tmp - st_BO_tmp)
            # optimize acqusition and get next observation
            x_next, y_next, acq_value, opt_ind, FOPs_time_tmp, BO_time_tmp = optimize_acq_and_get_observation(model, train_X, train_Y, nx, xL, xU, mFBA, solvername, i, mp_ind = mp_ind, n_p = n_p)
            FOPs_time = FOPs_time + FOPs_time_tmp
            BO_time = BO_time + BO_time_tmp
            if opt_ind:
                # calculate the test loss
                st_FOPs_tmp = time.time()
                y_test_next, _  = f_max(mFBA_test, x_next.numpy().tolist(), solvername = solvername, mp_ind = mp_ind, n_p = n_p)
                end_FOPs_tmp = time.time()
                FOPs_time = FOPs_time + (end_FOPs_tmp-st_FOPs_tmp)
                # append to data history
                x_history = numpy.append(x_history, x_next.numpy().reshape((1,-1)), axis=0)
                y_history = numpy.append(y_history, y_next.reshape((1,-1)), axis=0)
                y_test_history = numpy.append(y_test_history, y_test_next.reshape((1,-1)), axis=0)

                # update theta_est_best
                if numpy.max(y_history).item() == y_next.reshape((1,-1)):
                # if y_next.item()-best_observed_tmp >= 1e-8:
                    print("Yes, update the best values!!!")
                    theta_est_best = x_next.numpy().tolist() + [1-sum(x_next.numpy().tolist())]
                    best_observed_tmp = y_next.item()
                    best_observed_test_tmp = y_test_next.item()
                best_observed.append(best_observed_tmp) # be sure to update best value
                best_observed_test.append(best_observed_test_tmp) # be sure to update best value
                print("theta next ",[ '%.3f' % elem for elem in x_next.numpy().tolist() + [1-sum(x_next.numpy().tolist())]], "theta best ",[ '%.3f' % elem for elem in theta_est_best], "true theta ", theta_true)
                print("y next ",[ '%.6f' %y_next.item()])
                
                theta_est.append(theta_est_best)
                theta_l2_norm = np.linalg.norm(np.subtract(theta_est_best, theta_true))/(n_rxn-1)**0.5
                theta_l2_norm_lst.append(theta_l2_norm)
                
                # print the current best max
                print('Trial: %d|Iteration: %d|c loss: %.3f|Max training loss so far: %.6f|Max prediction loss so far: %.6f|Acquistion value %.3f'%(trial+1, i+1, theta_l2_norm, best_observed_tmp, best_observed_test_tmp, acq_value))
                bounds = torch.tensor([(xL[j], xU[j]) for j in range(nx)]).T
                if i+1 in PL_iter_lst:
                    # plot final prediction results, if possible
                    train_X = torch.tensor(x_history)
                    train_Y = torch.tensor(y_history)
                    model = train_model(train_X, train_Y, nu=nu_val)


                    # save GP model
                    torch.save(model.state_dict(), Base_path + '/model_state_trial=%s_iter=%s.pth'%(trial,i))
                    # save x_history and y_history to later build DKL model
                    with open(Base_path + '/x_history_trial=%s_iter=%s.npy' %(trial,i), 'wb') as f:
                        np.save(f, x_history)
                    with open(Base_path + '/y_history_trial=%s_iter=%s.npy' %(trial,i), 'wb') as f:
                        np.save(f, y_history)                

            else:
                best_observed.append(best_observed_tmp) # be sure to update best value
                best_observed_test.append(best_observed_test_tmp) # be sure to update best value
                theta_est.append(theta_est_best)
                theta_l2_norm = np.linalg.norm(np.subtract(theta_est_best, theta_true))/(n_rxn-1)**0.5
                theta_l2_norm_lst.append(theta_l2_norm)
                print("Not all instances solved to global optimal points.")
        # update the list of all trials
        best_observed_all.append(best_observed)
        best_observed_all_test.append(best_observed_test)

        theta_est_all.append(theta_est)
        theta_l2_norm_lst_all.append(theta_l2_norm_lst)


        # plot final prediction results, if possible
        train_X = torch.tensor(x_history)
        train_Y = torch.tensor(y_history)
        model = train_model(train_X, train_Y, nu=nu_val)
        et = time.time()
        total_time = et - st
        comp_time_lst.append(total_time)
        FOPs_time_lst.append(FOPs_time)
        BO_time_lst.append(BO_time)
        print(f"Trial {trial} completed in {total_time} s")


    # save the computation time results
    df_comp_time = pd.DataFrame()
    df_comp_time['Data seed'] = [data_seed for i in range(N_trial)]
    df_comp_time['Nexp'] = [n_exp for i in range(N_trial)]
    df_comp_time['Nrxn'] = [n_rxn for i in range(N_trial)]
    df_comp_time['Ntest'] = [n_test for i in range(N_trial)]
    df_comp_time['sigma'] = [sigma for i in range(N_trial)]
    df_comp_time['Niter'] = [N_iter for i in range(N_trial)]
    df_comp_time['Trial'] = [i for i in range(1,N_trial+1)]
    df_comp_time['Total time'] = comp_time_lst
    df_comp_time['BO time'] = BO_time_lst
    df_comp_time['FOPs time'] = FOPs_time_lst
    if mp_ind:
        df_comp_time['mp_ind'] = 1
    else:
        df_comp_time['mp_ind'] = 0



    # save the best data
    y_best = numpy.asarray(best_observed_all)
    y_test_best = numpy.asarray(best_observed_all_test)
    df = pd.DataFrame()
    for trial in range(N_trial):
        df["Y trial %s" %trial] = y_best[trial,:]
        df["Y_test trial %s" %trial] = y_test_best[trial,:]

    df["X true"] = [str(theta_true) for i in range(len(df))]
    for trial in range(N_trial):
        df["X trial %s" %trial] = theta_est_all[trial][:]
        df["theta loss trial %s" %trial] = theta_l2_norm_lst_all[trial][:]

    path = Base_path + '/BO_results.csv'

    df.to_csv(path)


    # post-processing 
    GLOBAL_MAXIMUM = neg_loss_true

    iters = numpy.arange(1, N_init + N_iter + 1)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.errorbar(iters, y_best.mean(axis=0), yerr=confidence_interval(y_best, N_trial), label="training (BO)", linewidth=1.5)
    ax.errorbar(iters, y_test_best.mean(axis=0), yerr=confidence_interval(y_test_best, N_trial), label="testing", linewidth=1.5)
    plt.plot([1, N_init+N_iter], [GLOBAL_MAXIMUM]*2, 'k', label="true best objective", linewidth=2)
    ax.set(xlabel='number of observations', ylabel='best objective value')
    ax.legend(loc='lower right');
    ax.set_ylim(bottom=-300, top =50)
    path =  Base_path + '/BO_learning_curves.png'
    plt.savefig(path)
    # append recorded computation time
    path =  'BO_results/computation_time.xlsx'
    # appending the data of df after the data of demo1.xlsx
    try:
        with pd.ExcelWriter(path,mode="a",engine="openpyxl",if_sheet_exists="overlay") as writer:
            df_comp_time.to_excel(writer, sheet_name="Sheet1",header=None, startrow=writer.sheets["Sheet1"].max_row,index=False)
    except:
        df_comp_time.to_excel(path,index=False)

def main():

    """
    wrapper code to run BO4IO on the FBA case study
    Execute the code in terminal:
    python BO4IO_FBA.py -nexp 50 -nrxn 2 -ntrial 5 -niter 250 -ninit 5 -n_data_seed 1 -init_seed 1 -n_p 10 -sigma 0.1
    """
    # Collect input for model parameter assignment.
    parser = argparse.ArgumentParser(description='Wrapper code to run BO4IO to learn cellular objectives in FBA')
    optional = parser._action_groups.pop()  # creates group of optional arguments
    required = parser.add_argument_group('required arguments')  # creates group of required arguments
    # required input
    # optional input
    optional.add_argument('-nexp', '--n_exp', help='number of exps', type=int, default = 10)
    optional.add_argument('-nrxn', '--n_rxn', help='number of rxns in the objective', type=int, default = 3)
    optional.add_argument('-ntrial', '--N_trial', help='number of trials', type=int, default = 5)
    optional.add_argument('-niter', '--N_iter', help='number of trials', type=int, default = 45)
    optional.add_argument('-ninit', '--N_init', help='number of initial samples', type=int, default = 5)
    optional.add_argument('-sigma', '--sigma', help='noise level', type=float, default = 0.0)
    optional.add_argument('-n_data_seed', '--n_data_seed', help='number of data seed to go through', type=int, default = 10)
    optional.add_argument('-init_seed', '--init_seed', help='initial data_seed', type=int, default = 1)
    optional.add_argument('-mp', '--mp', help='multiprocessing (1) or not (0)', type=int, default = 1)
    optional.add_argument('-n_p', '--n_p', help='number of processors', type=int, default = 32)

    parser._action_groups.append(optional)  # add optional values to the parser
    args = parser.parse_args()  # get the arguments from the program input, set them to args

    # Initialize the algorithm settings
    n_exp = args.n_exp
    n_rxn = args.n_rxn
    N_trial = args.N_trial
    N_iter = args.N_iter
    N_init = args.N_init
    solvername = 'gurobi'
    n_data_seed = args.n_data_seed
    init_seed = args.init_seed
    sigma = args.sigma

    # indicator for parallelization
    if args.mp == 1:
        mp_ind = True
    else:
        mp_ind = False
    n_p = args.n_p
    
    # make directories
    try:
        os.mkdir("BO_results")
    except:
        pass

    try:
        os.mkdir("BO_results/nrxn_%s"%n_rxn)
    except:
        pass

    # read ecoli model info file and create sets for pyomo models
    RSet,RSet_rev,RSet_irrev_pos,RSet_irrev_neg, MetSet, UB_dict, LB_dict, S_dict = read_model()
    # get redox potential objective's flux coefficient
    redox_rxn_tuple = get_redox_rxn_tuple(S_dict,RSet)

    for data_seed in range(init_seed,init_seed + n_data_seed):
        print("=======================Data Seed = %s===================" %data_seed)
        ### create train models
        # generate bounds
        UB_rand, LB_rand = generate_random_bounds(RSet,RSet_rev,RSet_irrev_pos,RSet_irrev_neg, UB_dict, LB_dict, numSam = 500, data_seed=data_seed)

        # create model
        mFBA, theta_true =generate_Ref_model_mp(n_exp, n_rxn, sigma,UB_rand, LB_rand, theta_seed = data_seed, noise_seed = data_seed)
        
        ### create test models
        # generate bounds
        UB_rand_test, LB_rand_test = generate_random_bounds(RSet,RSet_rev,RSet_irrev_pos,RSet_irrev_neg, UB_dict, LB_dict, numSam = 500, data_seed=100)

        # create model
        n_test = 50
        mFBA_test, theta_true_test =generate_Ref_model_mp(n_test, n_rxn, sigma,UB_rand_test, LB_rand_test, theta_seed = data_seed, noise_seed = 100, test_ind=True)
        
        print("============== Program Inputs/Settings ==============")
        print(f"Number of experiments = {n_exp}")
        print(f"Length of theta vector = {n_rxn}")
        print(f"Number of trials = {N_trial}")
        print(f"Number of iterations = {N_iter}")
        print(f"Number of initial samples = {N_init}")
        
        print("============== Initialization ==============")

        # define ground true loss
        x_true = theta_true
        st = time.time()
        neg_loss_true, opt_ind = f_max(mFBA, [i for i in x_true.tolist()[0:-1]], solvername = solvername, mp_ind = mp_ind, n_p=n_p)
        et = time.time()
        total_time = et - st
        print(f"The negative train_loss value given the true parameters is {neg_loss_true}")
        print(f"The CPU time needed to evaluate the loss was {total_time}")

        st = time.time()
        neg_loss_true, opt_ind = f_max(mFBA_test, [i for i in x_true.tolist()[0:-1]], solvername = solvername, mp_ind = mp_ind, n_p = n_p)
        et = time.time()
        total_time = et - st
        print(f"The negative test_loss value given the true parameters is {neg_loss_true}")
        print(f"The CPU time needed to evaluate the loss was {total_time}")

        # bounds on unknown parameters (theta)
        nx = n_rxn-1
        xL = numpy.array([0.01]*nx)
        xU = numpy.array([0.99]*nx)

        print("============== BO Loop ==============")

        st = time.time()
        # BO_Loop 
        BO_loop(mFBA, mFBA_test, n_exp, n_test, n_rxn, sigma, solvername, theta_true, neg_loss_true, nx, xL, xU, N_init = N_init, N_iter = N_iter, N_trial = N_trial, nu_val = 1.5, mp_ind = mp_ind, data_seed = data_seed, n_p = n_p)
        et = time.time()
        total_time = et - st
        print(f"The BO loop time needed was {total_time}")

if __name__ ==  '__main__':
    main()