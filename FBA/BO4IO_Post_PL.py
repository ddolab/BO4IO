from BO4IO_FBA import *
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from typing import Optional, Union
from torch import Tensor
from botorch.acquisition.objective import PosteriorTransform
from botorch.generation import get_best_candidates
from botorch.optim import gen_batch_initial_conditions
from Custom_Optimizers import gen_candidates_scipy_forPL

class LowerConfidenceBound_PL(AnalyticAcquisitionFunction):
    """
    Function was based on BoTorch UCB
    Analtical single-outcome Lower Confidence Bound (LCB).
    Create LCB for the GP posterior for the PL purpose

    LCB(x) = mu(x) - sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.
    """

    def __init__(
        self,
        model: Model,
        beta: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)
        return mean - self.beta.sqrt() * sigma#torch.min(mean - self.beta.sqrt() * sigma,0)[0] #flip sign because maximization 

class UpperConfidenceBound_PL(AnalyticAcquisitionFunction):
    """
    Function was based on BoTorch UCB
    Analtical single-outcome Upper Confidence Bound (UCB).   
    Create UCB for the GP posterior for the PL purpose

    UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> LCB = UpperConfidenceBound_PL(model, beta=0.2)
        >>> lcb = UCB(test_X)
    """

    def __init__(
        self,
        model: Model,
        beta: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            objective: A ScalarizedObjective (optional).
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b) x 1 x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X.to(torch.float64))
        # print("X shape: ", X.shape)
        # mean, sigma = self._mean_and_sigma(torch.flatten(X.to(torch.float64)))
        return mean + self.beta.sqrt() * sigma#torch.min(mean + self.beta.sqrt() * sigma,0)[0] #flip sign because maximization 



def l_star_optimization(objective, ic, bounds):
    """
    Find lstar for computing the finite-sample based confidence intervals
    """

    # specify inequality constraints of the form x(1) + ... + x(nx) <= 1 <-> (-1)*x(1) + ... + (-1)*x(nx) >= -1
    indices = torch.tensor([i for i in range(len(ic.squeeze(0)))])
    coefficients = torch.tensor([-1.0]*len(ic.squeeze(0)), dtype=torch.float64)
    rhs = -0.99
    inequality_constraints = [(indices, coefficients, rhs)]

    # multistarts settings
    N_starts = 10
    Xinit = gen_batch_initial_conditions(
    objective, bounds, q=1, num_restarts=N_starts, raw_samples=256,
    inequality_constraints = inequality_constraints)

    if not ic.isnan:
        Xinit = torch.cat((ic.unsqueeze(0),Xinit),0).squeeze(0)

    st = time.time()
    # optimization
    batch_candidates, batch_acq_values, res = gen_candidates_scipy_forPL(
    initial_conditions=Xinit,
    inequality_constraints = inequality_constraints,
    acquisition_function=objective,
    lower_bounds=bounds[0],
    upper_bounds=bounds[1],
    )

    # only report the results if the optimal solutions are returned
    if res.success:
        # get the best points
        theta_star = get_best_candidates(
            batch_candidates=batch_candidates, batch_values=batch_acq_values
        ).detach()
        end = time.time()
        l_star = objective(theta_star.squeeze(0).unsqueeze(0)).detach()
    else:
        l_star = torch.tensor(float('nan'))
        theta_star = torch.tensor([float('nan') for _ in range(len(ic.squeeze(0)))])
    return theta_star.squeeze(0), l_star

def ProfileLikelihoodApproximation(model, bounds, theta_best, l_best, n_rxn, all_range = True):
    """
    Profile likelihood approximation
    """
    
    # initialize sampled theta range (ratio)
    theta_range = 0.1 # percentage of perturbation from the theta_k^star
    n_PL = 1000 # number of the PL point sampled
    step_size = 1./n_PL
    
    
    st = time.time()
    # Define UCB and LCB of the current GP
    UCB = UpperConfidenceBound_PL(model, beta=1.96**2)
    LCB = LowerConfidenceBound_PL(model, beta=1.96**2)
    
    
    theta_best_LCB = model.posterior(theta_best).mean - 1.96* model.posterior(theta_best).stddev
    theta_best_UCB = model.posterior(theta_best).mean + 1.96* model.posterior(theta_best).stddev
    print("current_best: ", theta_best)
    print("current_best LCB: ", theta_best_LCB)
    print("current_best UCB: ", theta_best_UCB)
    
    
    
    # define IC (current best)
    ic = theta_best

    # get lstar_UCB
    theta_star_UCB, l_star_UCB = l_star_optimization(UCB, ic, bounds)
    print("theta_star_UCB: ", theta_star_UCB)
    print("l_star_UCB: ", l_star_UCB)

    # get lstar_LCB
    st = time.time()
    theta_star_LCB, l_star_LCB = l_star_optimization(LCB, ic, bounds)
    et = time.time()
    total_time = et - st
    print("theta_star_LCB: ", theta_star_LCB)
    print("l_star_LCB: ",  l_star_LCB)
    
    # initialize df for storing the PL data
    df_PL = pd.DataFrame()

    # outer-approximation (wosrt-case CI)
    print("========== OA starts ==========")
    for k in range(n_rxn-1):
        
        st = time.time()
        
        # upward from theta_k_star
        # initialize theta input in PL
        theta_k = theta_star_LCB[k].detach().clone()
        ic = theta_star_LCB.detach().clone()
        bounds_PL =  bounds.detach().clone()
        
        # initialize tmp df to store the PL results
        df_tmp = pd.DataFrame()
        theta_k_lst = []
        scaled_theta_k_lst = []
        theta_star_PL_lst = []
        l_PL_lst = []   

        # define sampling range
        if all_range:
            # full range [0.01,0.99]
            PL_LB = 0.01
            PL_UB = 1.00 - 0.01*(n_rxn-1)
        else:
            # range with specified ratio [theta_star*(1-theta_range), theta_star*(1+theta_range)]
            PL_LB = theta_k*(1-theta_range)
            PL_UB = theta_k*(1+theta_range)
    
        
        # initialize
        theta_k_tmp = theta_k
        
        # sampled PL at defined range
        while theta_k_tmp < PL_UB:

            # fix theta_k in bounds
            bounds_PL[0][k] = theta_k_tmp
            bounds_PL[1][k] = theta_k_tmp

            theta_star_tmp, l_star_tmp = l_star_optimization(UCB, ic.unsqueeze(0), bounds_PL)
            
            # store the results
            theta_k_lst.append(theta_k_tmp.item())
            theta_star_PL_lst.append(theta_star_tmp.detach().clone().tolist())
            l_PL_lst.append(l_star_tmp.item())
            scaled_theta_k_lst.append(theta_k_tmp.item()/theta_k.item())

            
            # update ic for the next sampling point
            ic = theta_star_tmp.detach().clone()
            
            # Next sampled theta_k value
            theta_k_tmp = theta_k_tmp + step_size


        # downward from theta_k_star
        # initialize theta input in PL
        theta_k = theta_star_LCB[k].detach().clone()
        ic = theta_star_LCB.detach().clone()
        bounds_PL =  bounds.detach().clone()
        
        # initialize
        theta_k_tmp = theta_k

        # sampled PL at defined range
        while theta_k_tmp > PL_LB:

            # fix theta_k in bounds
            bounds_PL[0][k] = theta_k_tmp
            bounds_PL[1][k] = theta_k_tmp
            theta_star_tmp, l_star_tmp = l_star_optimization(UCB, ic.unsqueeze(0), bounds_PL)
            
            # store the results
            theta_k_lst.append(theta_k_tmp.item())
            theta_star_PL_lst.append(theta_star_tmp.detach().clone().tolist())
            l_PL_lst.append(l_star_tmp.item())
            scaled_theta_k_lst.append(theta_k_tmp.item()/theta_k.item())

            # update ic for the next sampling point
            ic = theta_star_tmp.detach().clone()

            # Next sampled theta_k value
            theta_k_tmp = theta_k_tmp - step_size
        
        # store the results
        df_tmp['theta_k'] = theta_k_lst
        df_tmp['theta_ind'] = k+1
        df_tmp['scaled theta_k'] = scaled_theta_k_lst
        df_tmp['theta_star'] = theta_star_PL_lst
        df_tmp['l_PL'] = l_PL_lst
        df_tmp['l_star'] = [l_star_LCB.item()]*len(df_tmp)
        df_tmp['l_star_best_eval'] = [l_best]*len(df_tmp)
        df_tmp['theta_star_best_eval'] = [theta_best.detach().clone()[0].tolist()]*len(df_tmp)
        df_tmp['PL type'] = 'Outer Approx.'
        et = time.time()
        total_time = et - st
        print(f"theta {k} finished in {total_time} sec.")

        df_PL = pd.concat([df_PL,df_tmp], ignore_index=True)

    # inner-approximation (optimistic CI)
    print("========== IA starts ==========")
    for k in range(n_rxn-1):
        
        st = time.time()
        
        # upward from theta_k_star
        # initialize theta input in PL
        theta_k = theta_star_UCB[k].detach().clone()
        ic = theta_star_UCB.detach().clone()
        bounds_PL =  bounds.detach().clone()
        
        # initialize tmp df to store the PL results
        df_tmp = pd.DataFrame()
        theta_k_lst = []
        scaled_theta_k_lst = []
        theta_star_PL_lst = []
        l_PL_lst = []   


        # define sampling range
        if all_range:
            # full range [0.01,0.99]
            PL_LB = 0.01
            PL_UB = 1.00 - 0.01*(n_rxn-1)
        else:
            # range with specified ratio [theta_star*(1-theta_range), theta_star*(1+theta_range)]
            PL_LB = theta_k*(1-theta_range)
            PL_UB = theta_k*(1+theta_range)
        # initialize
        theta_k_tmp = theta_k

        # sampled PL at defined range
        while theta_k_tmp < PL_UB:

            # fix theta_k in bounds
            bounds_PL[0][k] = theta_k_tmp
            bounds_PL[1][k] = theta_k_tmp
            theta_star_tmp, l_star_tmp = l_star_optimization(LCB, ic.unsqueeze(0), bounds_PL)
            
            # store the results
            theta_k_lst.append(theta_k_tmp.item())
            theta_star_PL_lst.append(theta_star_tmp.detach().clone().tolist())
            l_PL_lst.append(l_star_tmp.item())
            scaled_theta_k_lst.append(theta_k_tmp.item()/theta_k.item())
            
            # update ic for the next sampling point
            ic = theta_star_tmp.detach().clone()

            # Next sampled theta_k value
            theta_k_tmp = theta_k_tmp + step_size

        # downward from theta_k_star
        # initialize theta input in PL
        theta_k = theta_star_UCB[k].detach().clone()
        ic = theta_star_UCB.detach().clone()
        bounds_PL =  bounds.detach().clone()
        
        # initialize
        theta_k_tmp = theta_k


        # sampled PL at defined range
        while theta_k_tmp > PL_LB:

            # fix theta_k in bounds
            bounds_PL[0][k] = theta_k_tmp
            bounds_PL[1][k] = theta_k_tmp
            theta_star_tmp, l_star_tmp = l_star_optimization(LCB, ic.unsqueeze(0), bounds_PL)
            
            # store the results
            theta_k_lst.append(theta_k_tmp.item())
            theta_star_PL_lst.append(theta_star_tmp.detach().clone().tolist())
            l_PL_lst.append(l_star_tmp.item())
            scaled_theta_k_lst.append(theta_k_tmp.item()/theta_k.item())


            # update ic for the next sampling point
            ic = theta_star_tmp.detach().clone()

            # Next sampled theta_k value
            theta_k_tmp = theta_k_tmp - step_size
        
        # store
        df_tmp['theta_k'] = theta_k_lst
        df_tmp['theta_ind'] = k+1
        df_tmp['scaled theta_k'] = scaled_theta_k_lst
        df_tmp['theta_star'] = theta_star_PL_lst
        df_tmp['l_PL'] = l_PL_lst
        df_tmp['l_star'] = [l_star_UCB.item()]*len(df_tmp)
        df_tmp['l_star_best_eval'] = [l_best]*len(df_tmp)
        df_tmp['theta_star_best_eval'] = [theta_best.detach().clone()[0].tolist()]*len(df_tmp)
        df_tmp['PL type'] = 'Inner Approx.'
        et = time.time()
        total_time = et - st
        print(f"theta {k} finished in {total_time} sec.")
    
    
        df_PL = pd.concat([df_PL,df_tmp], ignore_index=True)

    return df_PL

   
# @acqf_input_constructor(LowerConfidenceBound_PL)
# @acqf_input_constructor(UpperConfidenceBound_PL)

def identify_best_trial(df, N_iter, N_init, N_trial):
    """
    Pick the trial that shows the lowest prediction error to perform the PL analysis
    """
    # get testing loss among trials
    test_Y_lst = [df[f"Y_test trial {trial}"][N_iter+N_init-1] for trial in range(N_trial)]    

    # identify best trial
    best_trial = test_Y_lst.index(max(test_Y_lst))
    
    return best_trial

def main():
    """
    Wrapper code to compute profile likelihood
    Execute the code in terminal:
    python BO4IO_Post_PL.py -nexp 50 -nrxn 3 -ntrial 5 -niter 250 -ninit 5 -n_data_seed 1 -init_seed 1 -n_p 10 -sigma 0.1
    """
    # Collect input for model parameter assignment.
    parser = argparse.ArgumentParser(desCIiption='Wrapper code to compute profile likelihood')
    optional = parser._action_groups.pop()  # CIeates group of optional arguments
    required = parser.add_argument_group('required arguments')  # CIeates group of required arguments
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

    # optional.add_argument('-s', '--scen', help='linear (1) or nonlinear (2) objective', type=int, default = 1)

    parser._action_groups.append(optional)  # add optional values to the parser
    args = parser.parse_args()  # get the arguments from the program input, set them to args

    # Specify the algorithm settings
    n_exp = args.n_exp
    n_rxn = args.n_rxn
    N_trial = args.N_trial
    N_iter = args.N_iter
    N_init = args.N_init
    solvername = 'gurobi'
    n_data_seed = args.n_data_seed
    init_seed = args.init_seed
    sigma = args.sigma

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

    # read ecoli model info file and CIeate sets for pyomo models
    RSet,RSet_rev,RSet_irrev_pos,RSet_irrev_neg, MetSet, UB_dict, LB_dict, S_dict = read_model()
    # get redox potential objective's flux coefficient
    redox_rxn_tuple = get_redox_rxn_tuple(S_dict,RSet)
    st = time.time()
    for data_seed in range(init_seed,init_seed + n_data_seed):
        print("=======================Data Seed = %s===================" %data_seed)
        ### CIeate train models
        # generate bounds
        UB_rand, LB_rand = generate_random_bounds(RSet,RSet_rev,RSet_irrev_pos,RSet_irrev_neg, UB_dict, LB_dict, numSam = 500, data_seed=data_seed)

        # CIeate model
        mFBA, theta_true =generate_Ref_model_mp(n_exp, n_rxn, sigma,UB_rand, LB_rand, theta_seed = data_seed, noise_seed = data_seed)
        
        ### CIeate test models
        # generate bounds
        UB_rand_test, LB_rand_test = generate_random_bounds(RSet,RSet_rev,RSet_irrev_pos,RSet_irrev_neg, UB_dict, LB_dict, numSam = 500, data_seed=100)

        # CIeate model
        n_test = 50
        mFBA_test, theta_true_test =generate_Ref_model_mp(n_test, n_rxn, sigma,UB_rand_test, LB_rand_test, theta_seed = data_seed, noise_seed = 100, test_ind=True)
        
        print("============== Program Inputs/Settings ==============")
        print(f"Number of experiments = {n_exp}")
        print(f"Length of c vector = {n_rxn}")
        print(f"Number of trials = {N_trial}")
        print(f"Number of iterations = {N_iter}")
        print(f"Number of initial samples = {N_init}")
        
        print("============== Initialization ==============")
        print(theta_true)
        # construct test model

        # define ground true loss
        x_true = theta_true
        # print(x_true)
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
        bounds = torch.tensor([(xL[j], xU[j]) for j in range(nx)]).T


        base_path = "BO_results/nrxn_%s/sigma=%s_N_trial=%s_N_init=%s_N_iter=%s_n_exp=%s_n_rxn=%s_data_seed=%s"%(
            n_rxn,sigma,N_trial,N_init,N_iter,n_exp,n_rxn,data_seed
        )
        PL_iter_lst = [10,25,50,100,150,200,250]

        PL_iter_lst = [250]

        # read BO results
        df_BO = pd.read_csv(base_path + "/BO_results.csv")
        trial = identify_best_trial(df_BO, N_iter, N_init, N_trial)
        


        for trial in [trial]:

            for iter in PL_iter_lst:
                st_PL = time.time()

                print(f"================ Start PL at {iter} iteration (trail {trial})================")
                # initialize likelihood and model
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                
                # read X, Y data
                with open(base_path+'/y_history_trial=%s_iter=%s.npy'%(trial,iter-1), 'rb') as f:
                    y_history = np.load(f)
                with open(base_path+'/x_history_trial=%s_iter=%s.npy'%(trial,iter-1), 'rb') as f:
                    x_history = np.load(f)
                ### reCIeate GP surrogate
                # initialize model
                train_X = torch.tensor(x_history)
                train_Y = torch.tensor(y_history)
                model = train_model(train_X, train_Y, nu=1.5)
                # read the old GP hyperparameters and reCIeate the same model
                state_dict = torch.load(base_path+'/model_state_trial=%s_iter=%s.pth'%(trial,iter-1))
                model.load_state_dict(state_dict)   

                # # read best theta_est and l_star from df_BO
                theta_est_best =  ast.literal_eval(df_BO[f"X trial {trial}"][iter+N_init-1])[0:-1]
                best_observed = df_BO[f"Y trial {trial}"][iter+N_init-1]

                # read best theta_est and l_star from df_PL
                # theta_est_best =  ast.literal_eval(df_PL["theta_star_best_eval"][0])
                # best_observed = df_PL["l_star_best_eval"][0]
                
                # PL
                df_PL = ProfileLikelihoodApproximation(model, bounds, torch.FloatTensor([theta_est_best]),best_observed,n_rxn)
                path = base_path + '/BO_PL_trial=%s_iter=%s_post_n_10.csv'%(trial,iter-1)
                df_PL.to_csv(path,index=False)

                et_PL = time.time()
                print(f"PL takes {et_PL-st_PL} sec.")

    et = time.time()
    total_time = et - st
    print(f"The whole Post PL loop time needed was {total_time}")

if __name__ ==  '__main__':
    main()