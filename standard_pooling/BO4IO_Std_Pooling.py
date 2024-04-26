import pandas as pd
import ast
# Package Needed for BO4IO
import numpy
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import gurobipy as gp
import time
import botorch
import gpytorch
import numpy
import torch
import scipy
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints.constraints import Interval
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition import qExpectedImprovement
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
# new for composite EI
from botorch.models import SingleTaskGP, FixedNoiseGP, HigherOrderGP
from botorch.models.higher_order_gp import FlattenedStandardize
from linear_operator.settings import _fast_solves
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.sampling.normal import IIDNormalSampler
from torch.optim import Adam
from functools import partial
import gpytorch.settings as gpt_settings
from botorch.acquisition.objective import GenericMCObjective
from gpytorch.priors.torch_priors import GammaPrior, MultivariateNormalPrior
import sys

from botorch.optim import optimize_acqf
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import argparse
import os
import matplotlib.cbook
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import logging
import multiprocessing as mp
logging.getLogger('pyomo.core').setLevel(logging.CRITICAL)
# from simulator import *
from base_model.ref_paper_p_abstract import *
from data_read.poolsize import *
from data_read.product_quality_lower import *
from data_read.product_quality_upper import *
from data_read.pool_connections import *
import random


def recreate_instances(n_exp, sigma, filename, theta_seed, test_ind = False):
    """
    Recreate FOPs from the pre-synthesized datasets
    """
    # read the synthetic data file
    df = pd.read_excel(filename)
    data_0 = ast.literal_eval(df["Instance Data Dictionary"][0])
    if test_ind:
        df = df[::-1].reset_index(drop=True)

    x_history_dict = {i: [] for i in list(ast.literal_eval(df["Decision Variables"][0])['x'].keys())}
    y_history_dict = {i: [] for i in list(ast.literal_eval(df["Decision Variables"][0])['y'].keys())}
    z_history_dict = {i: [] for i in list(ast.literal_eval(df["Decision Variables"][0])['z'].keys())}
    p_history_dict = {i: [] for i in list(ast.literal_eval(df["Decision Variables"][0])['p'].keys())}
    
    # extract true theta values
    theta_UB_vector = ast.literal_eval(df["Upper True Theta Demand"][0])
    theta_UB_vector = list(theta_UB_vector.values())
    theta_LB_vector = ast.literal_eval(df["Lower True Theta Demand"][0])
    theta_LB_vector = list(theta_LB_vector.values())
    # extract actual demand
    actual_demand = [list(ast.literal_eval(df["Demand"][i]).values()) for i in range(len(df))]

    # recreate experimental conditions (models)
    model_lst = []
    actual_demand_tmp = []
    count_exp = 0
    n = 0

    xRef_dict = {i: [] for i in list(x_history_dict.keys())}
    yRef_dict = {i: [] for i in list(y_history_dict.keys())}
    pRef_dict = {i: [] for i in list(p_history_dict.keys())}

    np.random.seed(theta_seed)

    while count_exp < n_exp:
        data = ast.literal_eval(df["Instance Data Dictionary"][n])
        instance_p = model.create_instance(data)
        opt = pyo.SolverFactory('gurobi')#, solver_io="python")
        opt.options['NonConvex'] = 2
        opt.options['timelimit'] = 10
        opt.options["LogToConsole"]= 0
        opt.options["OutputFlag"]= 0
        opt.options['MIPGap'] = 0

        solver_res = opt.solve(instance_p, tee = False)#,keepfiles = True)
        ind_uniq = False
       
       
        if str(solver_res.solver.termination_condition) == 'optimal':

            actual_demand_tmp.append(actual_demand[n])
            count_exp += 1

            # set reference (ground-true) values of x, y, z, p 
            # set xRef
            count = 0
            for (i,l) in list(instance_p.Tx):
                instance_p.xRef[i, l] = pyo.value(instance_p.x[i, l])+np.random.normal(0, sigma)
                xRef_dict[i,l].append(pyo.value(instance_p.x[i, l]))
                count+=1


            # set yRef
            count = 0
            for l in list(instance_p.l):
                for j in list(instance_p.j):
                    instance_p.yRef[l,j] = pyo.value(instance_p.y[l,j])+np.random.normal(0, sigma)
                    yRef_dict[l,j].append(pyo.value(instance_p.y[l,j]))
                    count+=1
            # set pRef
            count = 0
            for l in list(instance_p.l):
                for k in list(instance_p.k):
                    instance_p.pRef[l,k] = pyo.value(instance_p.p[l,k])+np.random.normal(0, sigma)
                    pRef_dict[l,k].append(pyo.value(instance_p.p[l,k]))
                    count+=1
            
            # store the model
            model_lst.append(instance_p)
        n += 1
    actual_demand=actual_demand_tmp

    return model_lst, actual_demand, theta_UB_vector, theta_LB_vector

def simulator(m, actual_demand, theta_UB_vector, theta_LB_vector, selected_ind, n_exp, exp_id = 1, solvername = "ipopt", tee = False, mp_ind = False):
    """
    Solve FOPs with the randomized contextual inputs
    """
    for i, name in enumerate(selected_ind):
        # actual demand is always 1 here, so demand = theta
        m.Du[name+1]=theta_UB_vector[i]*actual_demand[exp_id][name] 
        
    opt = pyo.SolverFactory('gurobi')
    opt.options['NonConvex'] = 2
    opt.options["LogToConsole"]= 0
    opt.options["OutputFlag"]= 0
    opt.options['MIPGap'] = 0.0#1
    if mp_ind:
        opt.options['Threads'] = 4

    # indicator of global optimality
    opt_ind = False

    # if the problem is optimal or suboptimal, report the loos
    try:
        solver_res = opt.solve(m, tee = tee)#,keepfiles = True)
        # optimal
        if str(solver_res.solver.termination_condition) == 'optimal':
            loss = pyo.value(m.loss)
            opt_ind = True
        # suboptimal
        elif str(solver_res.solver.termination_condition) == 'maxTimeLimit':
            loss = pyo.value(m.loss)
        # infeasible
        else:
            print(str(solver_res.solver.termination_condition))
            loss = 10000
    except Exception as e:
        print(e)
        print("infeasible")       
        loss = 10000

    return loss, opt_ind

def f_max(model_lst, x, actual_demand,n_dim, selected_ind, solvername = 'gurobi', tee = False, mp_ind=False, n_p = 8):
    """
    Function to compute loss
    """    
    # update theta vector
    theta_Ref_LB = [0]*n_dim
    theta_Ref_UB = x

    n_exp = int(len(model_lst))
    
    if mp_ind:
        # solve FOPs in parallel
        p = mp.Pool(n_p)
        results = p.starmap_async(simulator, [(m, actual_demand, theta_Ref_UB, theta_Ref_LB, selected_ind,n_exp, id, solvername, tee, mp_ind) for id, m in enumerate(model_lst)])
        p.close()
        p.join()
        loss = [r[0] for r in results.get()]
        opt_ind = [r[1] for r in results.get()]
        
            

    else:
        # solve FOPs not in parallel
        loss = []
        opt_ind = []
        for id, m in enumerate(model_lst):
            loss_tmp, opt_ind_tmp = simulator(m, actual_demand, theta_Ref_UB, theta_Ref_LB, selected_ind,n_exp, id, solvername, tee, mp_ind)
            loss.append(loss_tmp)
            opt_ind.append(opt_ind_tmp)

    if all(opt_ind):
        # check if all FOPs are solve to global optimality
        opt_ind = True
    else:
        opt_ind = False
    loss = np.mean(loss)  
    return np.array(-loss), opt_ind


# functions for BO
# train model function
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
    fit_gpytorch_model(mll)
    # return the model
    return model


# need ability to optimize aquisition function
def optimize_one_step_acq(model, train_X, train_Y, xL, xU, nx, beta = 0.5, random_seed = 10):
    """
    Optimize the acquisition function to query candidate solutions
    """
    botorch.utils.sampling.manual_seed(seed=random_seed)
    torch.manual_seed(random_seed)
    optimize_acqf_kwargs = {
            "q": 1, #batch size
            "num_restarts": 10, #10
            "raw_samples": 512, #512
            "options": {"seed": random_seed}
        }


    acqf = UpperConfidenceBound(model, beta=4.0)
    # create hard bounds
    bounds = torch.tensor([(xL[j], xU[j]) for j in range(len(xL))]).T
    # find the next best theta
    numpy.random.seed(seed=random_seed)
    new_point_analytic, acq_value = optimize_acqf(acq_function=acqf, bounds=bounds, **optimize_acqf_kwargs)
    x_next = new_point_analytic[0,:]
    return x_next, acq_value

def optimize_acq_and_get_observation(model, model_lst, train_X, train_Y, actual_demand, xL, xU, n_dim, selected_ind, solvername, beta = 0.5, mp_ind = False, n_p = 8,random_seed = 10):
    """
    Wrapper to optimize acquisition funciton and perform loss calculation at the new query points
    """
    # run optimization to get next candidate design point
    x_next, acq_value = optimize_one_step_acq(model, train_X, train_Y, xL, xU, n_dim, beta = 0.5, random_seed = random_seed)
    # evaluate the true function at the next design
    y_next, opt_ind = f_max(model_lst, x_next.tolist(), actual_demand, n_dim, selected_ind, solvername = solvername, mp_ind = mp_ind, n_p = n_p)
    return x_next, y_next, acq_value, opt_ind

def main():
    """
    To executre the code, input the following lines in the command window.
    python BO4IO_Std_Pooling.py -nexp 50 -niter 100 -case_study haverly1 -ntrial 5 -sigma 0.05 -theta_seed 1 -n_data_seed 1 -mp 1 -n_p 25 -ntheta 2
    Input arguements:
    -nexp number of exps
    -ntrial number of trials
    -niter number of BO iterations
    -ninit number of initial samples
    -sigma noise level of ground-true data
    -case_study case study name
    """
    # Collect input for model parameter assignment.
    parser = argparse.ArgumentParser(description='BO4IO algorithm for standard pooling problems')
    optional = parser._action_groups.pop()  # creates group of optional arguments
    required = parser.add_argument_group('required arguments')  # creates group of required arguments
    # required input
    # required.add_argument('-nexp', '--n_exp', help='number of exps', type=int, required=True)
    # optional input
    optional.add_argument('-nexp', '--n_exp', help='number of exps', type=int, default = 25)
    optional.add_argument('-nobs', '--n_obs', help='number of observations of each exp', type=int, default = 10)
    optional.add_argument('-ntrial', '--N_trial', help='number of trials', type=int, default = 5)
    optional.add_argument('-niter', '--N_iter', help='number of BO iterations', type=int, default = 45)
    optional.add_argument('-ninit', '--N_init', help='number of initial samples', type=int, default = 5)
    optional.add_argument('-sigma', '--sigma', help='noise level', type=float, default = 0.0)
    optional.add_argument('-solvername', '--solvername', help='solvername', type=str, default = 'gurobi')
    optional.add_argument('-case_study', '--case_study', help='case study name', type=str, default = 'haverly1')
    optional.add_argument('-ntest', '--n_test', help='number of testing data', type=int, default = 50)
    optional.add_argument('-beta', '--beta', help='UCB Beta', type=float, default = 0.5)
    optional.add_argument('-nu', '--nu', help='matern kernel nu', type=float, default = 1.5)
    optional.add_argument('-n_data_seed', '--n_data_seed', help='number of data seed to go through', type=int, default = 1)
    optional.add_argument('-theta_seed', '--theta_seed', help='theta seed number', type=int, default = 1)
    optional.add_argument('-mp', '--mp', help='multiprocessing (1) or not (0)', type=int, default = 1)
    optional.add_argument('-n_p', '--n_p', help='number of processors', type=int, default = 32)
    optional.add_argument('-ntheta', '--ntheta', help='number of theta parameters to be estimated', type=int, default = 2)


    parser._action_groups.append(optional)  # add optional values to the parser
    args = parser.parse_args()  # get the arguments from the program input, set them to args
    
    # Initialize the algorithm settings
    n_exp = args.n_exp
    n_test = args.n_test
    N_trial =args.N_trial
    N_iter = args.N_iter
    N_init = args.N_init
    sigma = args.sigma
    solvername = "gurobi"
    case_study =  args.case_study
    beta =  args.beta
    nu = args.nu
    n_data_seed = args.n_data_seed
    theta_seed = args.theta_seed
    ntheta = args.ntheta
    # indicator for parallelization
    if args.mp == 1:
        mp_ind = True
        n_p = args.n_p
        np_test = args.n_p
    else:
        mp_ind = False
        n_p = 1
        np_test = 1
    
    # make directories
    try:
        os.mkdir("BO_results")
    except:
        pass

    try:
        os.mkdir("BO_results/case_study=%s"%case_study)
    except:
        pass

    try:
        os.mkdir("BO_results/case_study=%s/ntheta=%s"%(case_study, ntheta))
    except:
        pass



    # dimension of output streams
    n_dim = J[case_study]
    if n_dim < ntheta:
        print("The dimension of estimated theta (%s) is larger than the original problem (%s). Reduce the dimension."%(ntheta,n_dim))
        sys.exit()

    # generate the selected indexes of the theta vector
    theta_ind_lst = list(range(n_dim))
    random.seed(theta_seed)
    selected_ind = random.sample(theta_ind_lst,ntheta)


    # store information for PL at specified iterations
    PL_iter_lst = [10,25,50,100,150,200]


    for data_seed in range(theta_seed,theta_seed+n_data_seed):#range(1,n_data_seed+1):
        theta_seed = data_seed
        print("================== BO4IO-standard pooling starts=====================")
        print(f"case study {case_study}")
        print(f"sigma = {sigma}")
        print(f"n_exp = {n_exp}")
        print(f"seed = {theta_seed}")
        print(f"ntheta = {ntheta}")
        print(f"N_iter = {N_iter}")
        print(f"N_init = {N_init}")
        print(f"N_trial = {N_trial}")
        
        # create base path
        Base_path = 'BO_results/case_study=%s/ntheta=%s/case_study=%s_BO_N_trial=' %(case_study,ntheta,case_study) + \
          str(N_trial) + '_N_init=' + str(N_init) + '_N_iter=' +\
              str(N_iter) + '_n_exp=' + str(n_exp) + '_ntheta=' + \
                str(ntheta) + '_sigma=' + str(sigma) + \
                    '_nu=%s_beta=%s_theta_seed=%s_data_seed=%s' %(nu,beta,theta_seed,data_seed)
        try:
            os.mkdir(Base_path)
        except:
            pass
        # synthetic data filename
        filename = 'synthetic_data/%s_550_experiments_theta_seed=%s_data_seed=%s.xlsx'%(case_study,theta_seed, data_seed)

        # recreate instances from synthetic data
        # training instances
        st = time.time()
        model_lst, actual_demand, theta_UB_vector, theta_LB_vector = recreate_instances(n_exp, sigma, filename, theta_seed)
        et = time.time()
        print(f"The CPU time needed to recreate the instances was {et-st}")
        
        # testing instances
        model_lst_test, actual_demand_test, theta_UB_vector, theta_LB_vector = recreate_instances(n_test, sigma, filename, theta_seed, test_ind=True)
        theta_true_full = theta_UB_vector #+ theta_UB_vector


        


        print("True theta vector = ", theta_true_full)
        theta_true = [theta_true_full[i] for i in selected_ind]
        print("True theta vector (selected) = ", theta_true)

        # define ground true training loss
        st = time.time()
        neg_loss_true, _ = f_max(model_lst, theta_true, actual_demand, ntheta, selected_ind, mp_ind = mp_ind, n_p = n_p)
        et = time.time()
        total_time = et - st
        print(f"The negative training loss value given the true parameters is {neg_loss_true}")
        print(f"The CPU time needed to evaluate the loss was {total_time}")

        # define ground true testing loss
        st = time.time()
        neg_loss_true_test, _ = f_max(model_lst_test, theta_true, actual_demand_test, ntheta, selected_ind, mp_ind = mp_ind, n_p = np_test)
        et = time.time()
        total_time = et - st
        print(f"The negative testing loss value given the true parameters is {neg_loss_true_test}")
        print(f"The CPU time needed to evaluate the testing loss was {total_time}")

        # Define bounds of the estimated parameters (theta)
        nx = ntheta
        xL = numpy.array([0.5]*int(nx))# + [0.7]*int(nx/2))
        xU = numpy.array([1.0]*int(nx))# + [1.0]*int(nx/2))
            
        # BO loop
        # set value of hyperparameter in kernel
        nu_val = 1.5


        ### MAIN BO LOOP
        # initialize a list for best observed value
        best_observed_all = []
        best_observed_all_test = []
        theta_est_all = []
        theta_l2_norm_lst_all = []
        y_history_all = []
        x_history_all = []
        # create list to store computation time
        comp_ttime_lst = []
        # loop over number of trials
        for trial in range(N_trial):
            # record computation time
            st = time.time()

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

            # create empty list for theta_est (in the following c = x = theta, y = loss)
            theta_est = []
            theta_l2_norm_lst = []
            # loop over initial random samples
            init_count = 0
            while init_count < N_init:
                # intial samples
                x_next = [random.uniform(xL.tolist()[ii],xU.tolist()[ii]) for ii in range(ntheta)]
                x_next = np.array(x_next)
                st_FOP = time.time()
                y_next, opt_ind = f_max(model_lst, x_next.tolist(), actual_demand, ntheta, selected_ind, solvername = solvername, mp_ind = mp_ind, n_p = n_p)
                end_FOP = time.time()
                # print(f"FOPs solved in {end_FOP-st_FOP} sec")
                if opt_ind:
                    init_count += 1
                    st_FOP = time.time()
                    y_test_next,_ = f_max(model_lst_test, x_next.tolist(), actual_demand_test, ntheta, selected_ind, solvername = solvername, mp_ind = mp_ind, n_p = np_test)
                    end_FOP = time.time()
                    # print(f"Test set {_} solved in {end_FOP-st_FOP}")

                    x_history = numpy.append(x_history, x_next.reshape((1,-1)), axis=0)
                    y_history = numpy.append(y_history, y_next.reshape((1,-1)), axis=0)
                    y_test_history = numpy.append(y_test_history, y_test_next.reshape((1,-1)), axis=0)
                    
                    theta_est.append(x_next.tolist())
                    # update theta_est_best
                    if numpy.max(y_history).item() == y_next.reshape((1,-1)):
                        theta_est_best = x_next.tolist() #+ [1-sum(x_next.numpy().tolist())]
                        best_observed_tmp = y_next.item()#.reshape((1,-1))
                        best_observed_test_tmp = y_test_next.item()#.reshape((1,-1))
                    best_observed.append(best_observed_tmp) # be sure to update best value
                    best_observed_test.append(best_observed_test_tmp) # be sure to update best value
                    print("theta next ",[ '%.3f' % elem for elem in x_next], "theta best ",[ '%.3f' % elem for elem in theta_est_best], "true theta ", theta_true)

                    theta_l2_norm_lst.append(np.linalg.norm(np.subtract(theta_est_best, theta_true))/(nx)**0.5)

            # run main BO loop
            for i in range(N_iter):
                # convert data history to tensor
                train_X = torch.tensor(x_history)
                train_Y = torch.tensor(y_history)

                # call simple training function
                try:
                    model = train_model(train_X, train_Y, nu=nu)
                except Exception as e:
                    print(e)
                    print("Model training failed, so retained previous model")

                
                # optimize acqusition and get next observation
                x_next, y_next, acq_value, opt_ind = optimize_acq_and_get_observation(model, model_lst, train_X, train_Y, actual_demand, xL, xU, ntheta, selected_ind, solvername, beta = beta, mp_ind = mp_ind, n_p = n_p, random_seed = N_iter)

                if opt_ind:
                    # calculate the test loss
                    y_test_next,_ = f_max(model_lst_test, x_next.tolist(), actual_demand_test, ntheta, selected_ind, solvername = solvername, mp_ind = mp_ind, n_p = np_test)

                    # append to data history
                    x_history = numpy.append(x_history, x_next.numpy().reshape((1,-1)), axis=0)
                    y_history = numpy.append(y_history, y_next.reshape((1,-1)), axis=0)
                    y_test_history = numpy.append(y_test_history, y_test_next.reshape((1,-1)), axis=0)
                    
                    # update theta_est_best
                    if numpy.max(y_history).item() == y_next.reshape((1,-1)):
                        print("Yes, update the best values!!!")
                        theta_est_best = x_next.numpy().tolist()
                        best_observed_tmp = y_next.item()
                        best_observed_test_tmp = y_test_next.item()
                    best_observed.append(best_observed_tmp) # be sure to update best value
                    best_observed_test.append(best_observed_test_tmp) # be sure to update best value
                    
                    theta_est.append(theta_est_best)
                    theta_l2_norm = np.linalg.norm(np.subtract(theta_est_best, theta_true))/(nx)**0.5
                    theta_l2_norm_lst.append(theta_l2_norm)
                    # print the current best max
                    print("============= Traditional BO ================")
                    print('Trial: %d|Iteration: %d|c loss: %.4f|Max training value so far: %.4f|Max testing value so far: %.4f|Acquistion value %.4f'%(trial+1, i+1, theta_l2_norm, best_observed_tmp, best_observed_test_tmp, acq_value))
                    print(f"y next {y_next.item()}", "theta next ",[ '%.3f' % elem for elem in x_next], "theta best ",[ '%.3f' % elem for elem in theta_est_best], "true theta ", theta_true)

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
                    theta_l2_norm = np.linalg.norm(np.subtract(theta_est_best, theta_true))/(nx)**0.5
                    theta_l2_norm_lst.append(theta_l2_norm)
                    print("Not all instances solved to global optimal points.")


            # update the list of all trials
            best_observed_all.append(best_observed)
            best_observed_all_test.append(best_observed_test)
            theta_est_all.append(theta_est)
            theta_l2_norm_lst_all.append(theta_l2_norm_lst)
            y_history_all.append(y_history)
            x_history_all.append(x_history)

            train_X = torch.tensor(x_history)
            train_Y = torch.tensor(y_history)
            model = train_model(train_X, train_Y, nu=nu)
            et = time.time()
            total_time = et - st
            comp_ttime_lst.append(total_time)
            print(f"Trial {trial} completed in {total_time} s")
        

        # save the computation time results
        df_comp_time = pd.DataFrame()
        df_comp_time['Case study'] = [case_study for i in range(N_trial)]
        df_comp_time['Ntrain'] = [n_exp for i in range(N_trial)]
        df_comp_time['Ntest'] = [n_test for i in range(N_trial)]
        df_comp_time['theta_seed'] = [theta_seed for i in range(N_trial)]
        df_comp_time['data_seed'] = [data_seed for i in range(N_trial)]
        df_comp_time['sigma'] = [sigma for i in range(N_trial)]
        df_comp_time['Niter'] = [N_iter for i in range(N_trial)]
        df_comp_time['Trial'] = [i for i in range(1,N_trial+1)]
        df_comp_time['Computation time'] = comp_ttime_lst


        

        # save the best data
        try:
            os.mkdir('BO_results/')
        except:
            pass

        y_best = numpy.asarray(best_observed_all)
        y_test_best = numpy.asarray(best_observed_all_test)
        y_hitory_lst = numpy.asarray(y_history_all)
        x_hitory_lst = numpy.asarray(x_history_all)

        df = pd.DataFrame()
        for trial in range(N_trial):
            df["Y trial %s" %trial] = y_best[trial,:]
            df["Y_test trial %s" %trial] = y_test_best[trial,:]

        df["X true"] = [str(theta_true) for i in range(len(df))]
        for trial in range(N_trial):
            df["X trial %s" %trial] = theta_est_all[trial][:]
            df["theta loss trial %s" %trial] = theta_l2_norm_lst_all[trial][:]
        df["neg_loss_true"] = neg_loss_true
        df["neg_loss_true_test"] = neg_loss_true_test
        path = Base_path + "/BO_results.csv"
        df.to_csv(path)


        def confidence_interval(y):
            return 1.96 * y.std(axis=0) / numpy.sqrt(N_trial)

        GLOBAL_MAXIMUM = neg_loss_true

        iters = numpy.arange(1, N_init + N_iter + 1)
        y_best = numpy.asarray(best_observed_all)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.errorbar(iters, y_best.mean(axis=0), yerr=confidence_interval(y_best), label="BO", linewidth=2.5)
        plt.plot([1, N_init+N_iter], [GLOBAL_MAXIMUM]*2, 'k', label="true best objective", linewidth=2)
        ax.set(xlabel='number of observations', ylabel='best objective value')
        ax.legend(loc='lower right');
        path =  Base_path + '/BO_learning_curves.png'
        plt.savefig(path)

        # append recorded computation time
        path = 'BO_results/computation_time.xlsx'
        # appending the data of df after the data of computation_time.xlsx
        try:
            with pd.ExcelWriter(path,mode="a",engine="openpyxl",if_sheet_exists="overlay") as writer:
                df_comp_time.to_excel(writer, sheet_name="Sheet1",header=None, startrow=writer.sheets["Sheet1"].max_row,index=False)
        except:
            df_comp_time.to_excel(path,index=False)

if __name__ ==  '__main__':
    main()
