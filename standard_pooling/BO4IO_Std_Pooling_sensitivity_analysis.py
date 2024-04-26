from BO4IO_Std_Pooling import *

def main():
    """
    Perform sensitivity analysis on the specified case study
    python BO4IO_Std_Pooling_sensitivity_analysis.py -case_study haverly1  -ntheta 2 -nexp 50 -nsteps 500 -mp 1 -n_p 10 -theta_seed 1 -n_data_seed 1 -sigma 0.05
    """
    # Collect input for model parameter assignment.
    parser = argparse.ArgumentParser(description='Perform sensitivity analysis on the specified case study')
    optional = parser._action_groups.pop()  # creates group of optional arguments
    required = parser.add_argument_group('required arguments')  # creates group of required arguments
    # required input
    # optional input
    optional.add_argument('-nexp', '--n_exp', help='number of exps', type=int, default = 25)
    optional.add_argument('-nsteps', '--N_steps', help='number of testing steps in sensitivity analysis', type=int, default = 5)
    optional.add_argument('-ntrial', '--N_trial', help='number of trials', type=int, default = 5)
    optional.add_argument('-niter', '--N_iter', help='number of BO iterations', type=int, default = 45)
    optional.add_argument('-ninit', '--N_init', help='number of initial samples', type=int, default = 5)
    optional.add_argument('-sigma', '--sigma', help='noise level', type=float, default = 0.0)
    optional.add_argument('-solvername', '--solvername', help='solvername', type=str, default = 'gurobi')
    optional.add_argument('-case_study', '--case_study', help='case study name', type=str, default = 'Lee')
    optional.add_argument('-ntest', '--n_test', help='number of testing data', type=int, default = 10)
    optional.add_argument('-beta', '--beta', help='UCB Beta', type=float, default = 0.5)
    optional.add_argument('-nu', '--nu', help='matern kernel nu', type=float, default = 1.5)
    optional.add_argument('-n_data_seed', '--n_data_seed', help='number of data seed to go through', type=int, default = 1)
    optional.add_argument('-theta_seed', '--theta_seed', help='theta seed number', type=int, default = 1)
    optional.add_argument('-gap', '--gap', help='If sepecify gap for SI. Yes = 1, No = 0', type=int, default = 0)
    optional.add_argument('-mp', '--mp', help='multiprocessing (1) or not (0)', type=int, default = 1)
    optional.add_argument('-n_p', '--n_p', help='number of processors', type=int, default = 32)
    optional.add_argument('-ntheta', '--ntheta', help='number of theta parameters to be estimated', type=int, default = 2)


    parser._action_groups.append(optional)  # add optional values to the parser
    args = parser.parse_args()  # get the arguments from the program input, set them to args
    
    # Specify the algorithm settings
    n_exp = args.n_exp
    n_test = args.n_test
    N_steps = args.N_steps
    N_trial =args.N_trial
    N_iter = args.N_iter
    N_init = args.N_init
    sigma = args.sigma
    solvername = "gurobi"#args.solvername
    case_study =  args.case_study
    beta =  args.beta
    ntheta = args.ntheta
    nu = args.nu
    n_data_seed = args.n_data_seed
    theta_seed = args.theta_seed
    ntheta = args.ntheta
    if args.mp == 1:
        mp_ind = True
        n_p = args.n_p

    else:
        mp_ind = False
        n_p = 1

    try:
        os.mkdir("sensitivity_analysis/")
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
    for data_seed in range(theta_seed,n_data_seed+theta_seed):
        # synthetic data filename
        filename = 'synthetic_data/%s_550_experiments_theta_seed=%s_data_seed=%s.xlsx'%(case_study, data_seed,data_seed)
        with gp.Env() as env:
            # recreate instances from synthetic data
            # training instances
            st = time.time()
            model_lst, actual_demand, theta_UB_vector, theta_LB_vector = recreate_instances(n_exp, sigma, filename, data_seed)
            et = time.time()
            print(f"The CPU time needed to recreate the instances was {et-st}")
            
            theta_true_full = theta_UB_vector #+ theta_UB_vector

            print("True theta vector = ", theta_true_full)

            theta_true = [theta_true_full[i] for i in selected_ind]
            print("True theta vector (selected) = ", theta_true)

            # define ground true training loss
            st = time.time()
            neg_loss_true, _ = f_max(model_lst, theta_true, actual_demand, ntheta, selected_ind, mp_ind = mp_ind, n_p = n_p)
            # print(theta_true,theta_test)
            et = time.time()
            total_time = et - st
            print(f"The negative training loss value given the true parameters is {neg_loss_true}")
            print(f"The CPU time needed to evaluate the loss was {total_time}")

            nx = ntheta

            # initialize SI range
            
            df_SI = pd.DataFrame()
            df_SI_true = pd.DataFrame()
            df_SI_true['theta_true'] = theta_true
            df_SI_true['loss'] = neg_loss_true
            df_SI_true['type'] = ["theta_%s_true"%(i+1) for i in range(ntheta)]
            # SI on lower bounds
            for i in range(ntheta):
                if args.gap == 1:
                    x_lst = np.linspace(xL[i],xU[i],N_steps, endpoint=True).tolist()
                else:
                    x_lst = np.linspace(0.5,1.0,N_steps, endpoint=True).tolist()
                df_tmp = pd.DataFrame()

                x_history = []
                y_history = []
                for x_tmp in x_lst:
                    theta_tmp = theta_true.copy()
                    theta_tmp[i] = x_tmp
                    loss_tmp,_ = f_max(model_lst, theta_tmp, actual_demand, ntheta, selected_ind, mp_ind = mp_ind, n_p = n_p)
                    
                    x_history.append(x_tmp)
                    y_history.append(loss_tmp.item())
                
                # theta true
                loss_tmp,_ = f_max(model_lst, theta_true, actual_demand, ntheta, selected_ind, mp_ind = mp_ind, n_p = n_p)
                x_history.append(theta_true[i])
                y_history.append(loss_tmp.item())

                # store the SI results
                df_tmp['theta'] = x_history
                df_tmp['loss'] = y_history
                df_tmp['type'] = "theta_%s"%(i+1)
                df_SI = pd.concat([df_SI,df_tmp], ignore_index=True)
            
            fig, ax = plt.subplots()
            sns.lineplot(ax=ax,data=df_SI, x="theta", y="loss", hue="type")
            sns.scatterplot(ax=ax,data=df_SI_true, x="theta_true", y="loss", hue="type")
            ax.legend(frameon=False)
            fig.savefig("sensitivity_analysis/SI_%s_n_exp=%s_ntheta=%s_n=%s_gap=%s_seed=%s.png"%(case_study,n_exp, ntheta,N_steps,args.gap,data_seed))
            df_SI.to_csv("sensitivity_analysis/SI_%s_n_exp=%s_ntheta=%s_n=%s_gap=%s_seed=%s.csv"%(case_study,n_exp, ntheta,N_steps,args.gap,data_seed))


if __name__ ==  '__main__':
    main()
