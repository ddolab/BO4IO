from BO4IO_Gen_Pooling import *
from botorch.models.model import Model
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from typing import Optional, Union
from torch import Tensor
from botorch.acquisition.objective import PosteriorTransform

from botorch.acquisition.input_constructors import acqf_input_constructor

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

@acqf_input_constructor(LowerConfidenceBound_PL)
@acqf_input_constructor(UpperConfidenceBound_PL)

def l_star_optimization(objective, ic, bounds):
    """
    Find lstar for computing the finite-sample based confidence intervals
    """

    # multistarts settings
    N_starts = 10
    Xinit = gen_batch_initial_conditions(
    objective, bounds, q=1, num_restarts=N_starts, raw_samples=256,
    )

    if not ic.isnan:
        Xinit = torch.cat((ic.unsqueeze(0),Xinit),0).squeeze(0)


    st = time.time()
    # optimization
    batch_candidates, batch_acq_values, res = gen_candidates_scipy_forPL(
    initial_conditions=Xinit, 
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

def ProfileLikelihoodApproximation(model, bounds, theta_best, l_best, ntheta, all_range = True):
    """
    Profile likelihood approximation
    """
    # initialize sampled theta range (ratio)
    theta_range = 0.1 # percentage of perturbation from the theta_k^star
    n_PL = 400 # number of the PL point sampled
    step_size = 0.4/n_PL
    
    
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

    # outer-approximation (wosrt-case CR)
    print("========== OA starts ==========")
    for k in range(ntheta):
        
        st = time.time()
        
        # upward from theta_k_star
        # initialize theta input in PL
        # theta_k = theta_star_LCB[k].detach().clone()
        # ic = theta_star_LCB.detach().clone()
        theta_k = theta_best.squeeze(0)[k].detach().clone()
        ic = theta_best.squeeze(0).detach().clone()
        bounds_PL =  bounds.detach().clone()
        
        # initialize tmp df to store the PL results
        df_tmp = pd.DataFrame()
        theta_k_lst = []
        scaled_theta_k_lst = []
        theta_star_PL_lst = []
        l_PL_lst = []   

        # define sampling range
        if all_range:
            # full range [0.2,0.6]
            PL_LB = 0.2
            PL_UB = 0.6
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
            # scaled_theta_k_lst.append(theta_k_tmp.item()/theta_k.item())
            scaled_theta_k_lst.append(theta_k_tmp.item())

            
            # update ic for the next sampling point
            ic = theta_star_tmp.detach().clone()
            
            # Next sampled theta_k value
            theta_k_tmp = theta_k_tmp + step_size


        # downward from theta_k_star
        # initialize theta input in PL
        # theta_k = theta_star_LCB[k].detach().clone()
        # ic = theta_star_LCB.detach().clone()
        theta_k = theta_best.squeeze(0)[k].detach().clone()
        ic = theta_best.squeeze(0).detach().clone()
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
            # scaled_theta_k_lst.append(theta_k_tmp.item()/theta_k.item())
            scaled_theta_k_lst.append(theta_k_tmp.item())

            # update ic for the next sampling point
            ic = theta_star_tmp.detach().clone()

            # Next sampled theta_k value
            theta_k_tmp = theta_k_tmp - step_size

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
    for k in range(ntheta):
        
        st = time.time()
        
        # upward from theta_k_star
        # initialize theta input in PL
        # theta_k = theta_star_UCB[k].detach().clone()
        # ic = theta_star_UCB.detach().clone()
        theta_k = theta_best.squeeze(0)[k].detach().clone()
        ic = theta_best.squeeze(0).detach().clone()
        bounds_PL =  bounds.detach().clone()
        
        # initialize tmp df to store the PL results
        df_tmp = pd.DataFrame()
        theta_k_lst = []
        scaled_theta_k_lst = []
        theta_star_PL_lst = []
        l_PL_lst = []   


        # define sampling range
        if all_range:
            # full range [0.2,0.6]
            PL_LB = 0.2
            PL_UB = 0.6
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
            # scaled_theta_k_lst.append(theta_k_tmp.item()/theta_k.item())
            scaled_theta_k_lst.append(theta_k_tmp.item())
            
            # update ic for the next sampling point
            ic = theta_star_tmp.detach().clone()

            # Next sampled theta_k value
            theta_k_tmp = theta_k_tmp + step_size

        # downward from theta_k_star
        # initialize theta input in PL
        # theta_k = theta_star_UCB[k].detach().clone()
        # ic = theta_star_UCB.detach().clone()
        theta_k = theta_best.squeeze(0)[k].detach().clone()
        ic = theta_best.squeeze(0).detach().clone()
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
            # scaled_theta_k_lst.append(theta_k_tmp.item()/theta_k.item())
            scaled_theta_k_lst.append(theta_k_tmp.item())


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
        df_tmp['l_star'] = [l_star_UCB.item()]*len(df_tmp)
        df_tmp['l_star_best_eval'] = [l_best]*len(df_tmp)
        df_tmp['theta_star_best_eval'] = [theta_best.detach().clone()[0].tolist()]*len(df_tmp)
        df_tmp['PL type'] = 'Inner Approx.'
        et = time.time()
        total_time = et - st
        print(f"theta {k} finished in {total_time} sec.")
    
    
        df_PL = pd.concat([df_PL,df_tmp], ignore_index=True)

    return df_PL

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
    python BO4IO_PostPL.py -nexp 50 -niter 200 -case_study Lee -ntrial 5 -sigma 0.05 -theta_seed 1 -n_data_seed 1 -mp 1 -n_p 10
    """
    # Collect input for model parameter assignment.
    parser = argparse.ArgumentParser(description='Wrapper code to compute profile likelihood')
    optional = parser._action_groups.pop()  # creates group of optional arguments
    required = parser.add_argument_group('required arguments')  # creates group of required arguments
    # required input
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
    
    # Specify the algorithm settings
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
    
    if args.mp == 1:
        mp_ind = True
        n_p = args.n_p
        np_test = 10
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
    # dimension of output streams
    n_dim = J
    # store information for PL at specified iterations
    PL_iter_lst = [10,25,50,100,150,200]#,500]

    for data_seed in range(theta_seed,theta_seed+n_data_seed):#range(1,n_data_seed+1):
        theta_seed = data_seed
        print("================== BO4IO-generalized pooling PL starts=====================")
        print(f"case study {case_study}")
        print(f"sigma = {sigma}")
        print(f"n_exp = {n_exp}")
        print(f"seed = {theta_seed}")
        print(f"ntheta = {n_dim}")
        print(f"N_iter = {N_iter}")
        print(f"N_init = {N_init}")
        print(f"N_trial = {N_trial}")
        
        
        
        
        Base_path = 'BO_results/case_study=%s/case_study=%s_BO_N_trial=' %(case_study,case_study) + \
          str(N_trial) + '_N_init=' + str(N_init) + '_N_iter=' +\
              str(N_iter) + '_n_exp=' + str(n_exp) + \
               '_sigma=' + str(sigma) + \
                    '_nu=%s_beta=%s_theta_seed=%s_data_seed=%s' %(nu,beta,theta_seed,data_seed)
        try:
            os.mkdir(Base_path)
        except:
            pass
        # synthetic data filename
        filename = 'synthetic_data/Lee_550_experiments_data_seed=%s_theta_seed=%s.xlsx'%(data_seed,theta_seed)

        # recreate instances from synthetic data
        # training instances
        model_lst, theta_vector,theta_UB, theta_LB = recreate_instances(n_exp, sigma, filename,data_seed)
        print("True theta vector = ", theta_vector)
        # testing instances
        model_lst_test, theta_vector,theta_UB_test, theta_LB_test = recreate_instances(n_test, sigma, filename,data_seed, test_ind=True)
        theta_true = theta_vector

        # define ground true training loss
        st = time.time()
        neg_loss_true, _ = f_max(model_lst, theta_true, n_dim, mp_ind = mp_ind, n_p = n_p)
        et = time.time()
        total_time = et - st
        print(f"The negative training loss value given the true parameters is {neg_loss_true}")
        print(f"The CPU time needed to evaluate the loss was {total_time}")

        # define ground true testing loss
        st = time.time()
        neg_loss_true_test, _ = f_max(model_lst_test, theta_true, n_dim, mp_ind = mp_ind, n_p = n_p)
        et = time.time()
        total_time = et - st
        print(f"The negative testing loss value given the true parameters is {neg_loss_true_test}")
        print(f"The CPU time needed to evaluate the testing loss was {total_time}")

        # Define bounds of the estimated parameters (theta)
        nx = n_dim
        xL = numpy.array([0.2]*int(nx))# + [0.7]*int(nx/2))
        xU = numpy.array([0.6]*int(nx))# + [1.0]*int(nx/2))
        bounds = torch.tensor([(xL[j], xU[j]) for j in range(nx)]).T

        # list of the BO iterations that need the PL analysis
        PL_iter_lst = [10,25,50,100,150,200]

        # read BO results
        df_BO = pd.read_csv(Base_path + "/BO_results.csv")
        trial = identify_best_trial(df_BO, N_iter, N_init, N_trial)

        for trial in [trial]:
            for iter in PL_iter_lst:
                st_PL = time.time()

                print(f"================ Start PL at {iter} iteration (trail {trial})================")
                # initialize likelihood and model
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                
                # read X, Y data
                with open(Base_path+'/y_history_trial=%s_iter=%s.npy'%(trial,iter-1), 'rb') as f:
                    y_history = np.load(f)
                with open(Base_path+'/x_history_trial=%s_iter=%s.npy'%(trial,iter-1), 'rb') as f:
                    x_history = np.load(f)
                ### recreate GP surrogate
                # initialize model
                train_X = torch.tensor(x_history)
                train_Y = torch.tensor(y_history)
                model = train_model(train_X, train_Y, nu=1.5)
                # read the old GP hyperparameters and recreate the same model
                state_dict = torch.load(Base_path+'/model_state_trial=%s_iter=%s.pth'%(trial,iter-1))
                model.load_state_dict(state_dict)   

                # # read best c_est and l_star from df_BO
                theta_est_best =  ast.literal_eval(df_BO[f"X trial {trial}"][iter+N_init-1])
                best_observed = df_BO[f"Y trial {trial}"][iter+N_init-1]

                # PL
                df_PL = ProfileLikelihoodApproximation(model, bounds, torch.FloatTensor([theta_est_best]),best_observed,nx)
                path = Base_path + '/BO_PL_trial=%s_iter=%s_post_n_10_currentbest.csv'%(trial,iter-1)
                df_PL.to_csv(path,index=False)

                et_PL = time.time()
                print(f"PL takes {et_PL-st_PL} sec.")
    
    et = time.time()
    total_time = et - st
    print(f"The whole Post PL loop time needed was {total_time}")

if __name__ ==  '__main__':
    main()