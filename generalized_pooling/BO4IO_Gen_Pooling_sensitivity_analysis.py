from BO4IO_Gen_Pooling import *
import seaborn as sns
def main():
    """
    Perform sensitivity analysis on the specified case study
    python BO4IO_Gen_Pooling_sensitivity_analysis.py -case_study Lee  -nexp 50 -nsteps 500 -mp 1 -n_p 10 -theta_seed 1 -n_data_seed 1 -sigma 0.05
    """
    # Collect input for model parameter assignment.
    parser = argparse.ArgumentParser(description='Perform sensitivity analysis on the specified case study')
    optional = parser._action_groups.pop()  # creates group of optional arguments
    required = parser.add_argument_group('required arguments')  # creates group of required arguments
    # required input
    # optional input
    optional.add_argument('-nexp', '--n_exp', help='number of exps', type=int, default = 25)
    optional.add_argument('-nsteps', '--N_steps', help='number of testing steps in sensitivity analysis', type=int, default = 5)
    optional.add_argument('-sigma', '--sigma', help='noise level', type=float, default = 0.0)
    optional.add_argument('-solvername', '--solvername', help='solvername', type=str, default = 'gurobi')
    optional.add_argument('-case_study', '--case_study', help='case study name', type=str, default = 'Lee')
    optional.add_argument('-ntest', '--n_test', help='number of testing data', type=int, default = 50)
    optional.add_argument('-beta', '--beta', help='UCB Beta', type=float, default = 0.5)
    optional.add_argument('-nu', '--nu', help='matern kernel nu', type=float, default = 1.5)
    optional.add_argument('-n_data_seed', '--n_data_seed', help='number of data seed to go through', type=int, default = 1)
    optional.add_argument('-theta_seed', '--theta_seed', help='theta seed number', type=int, default = 1)
    optional.add_argument('-gap', '--gap', help='If sepecify gap for SI. Yes = 1, No = 0', type=int, default = 0)
    optional.add_argument('-mp', '--mp', help='multiprocessing (1) or not (0)', type=int, default = 1)
    optional.add_argument('-n_p', '--n_p', help='number of processors', type=int, default = 32)


    parser._action_groups.append(optional)  # add optional values to the parser
    args = parser.parse_args()  # get the arguments from the program input, set them to args
    
    # Specify the algorithm settings
    n_exp = args.n_exp
    n_test = args.n_test
    N_steps = args.N_steps
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

    else:
        mp_ind = False
        n_p = 1

    try:
        os.mkdir("sensitivity_analysis/")
    except:
        pass

    for data_seed in range(theta_seed,n_data_seed+theta_seed):
        # synthetic data filename
        filename = 'synthetic_data/Lee_550_experiments_data_seed=%s_theta_seed=%s.xlsx'%(data_seed,data_seed)

        # recreate instances from synthetic data
        # training instances
        model_lst, theta_vector,theta_UB, theta_LB = recreate_instances(n_exp, sigma, filename,data_seed)
        # testing instances
        model_lst_test, theta_vector,theta_UB_test, theta_LB_test = recreate_instances(n_test, sigma, filename,data_seed, test_ind=True)
        theta_true = theta_vector

        # dimension of output streams
        n_dim = J

        print("True theta vector = ", theta_true)

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
        # xL = numpy.array([0.2]*int(nx))
        # xU = numpy.array([0.6]*int(nx))
        x_gap = 0.1
        xL = numpy.array([i-x_gap/2 for i in theta_true])
        xU = numpy.array([i+x_gap/2 for i in theta_true])

        # initialize SI range
        

        df_SI = pd.DataFrame()
        df_SI_true = pd.DataFrame()
        df_SI_true['theta_true'] = theta_true
        df_SI_true['loss'] = neg_loss_true
        df_SI_true['type'] = ["theta_%s_true"%(i+1) for i in range(n_dim)]
        # SI on lower bounds
        for i in range(n_dim):
            if args.gap == 1:
                x_lst = np.linspace(xL[i],xU[i],N_steps, endpoint=True).tolist()
            else:
                x_lst = np.linspace(0.2,0.6,N_steps, endpoint=True).tolist()
            df_tmp = pd.DataFrame()

            x_history = []
            y_history = []
            for xL_tmp in x_lst:
                theta_tmp = theta_true.copy()
                theta_tmp[i] = xL_tmp
                loss_tmp,_ = f_max(model_lst, theta_tmp, n_dim, mp_ind = mp_ind, n_p = n_p)
                x_history.append(xL_tmp)
                y_history.append(loss_tmp.item())
            
            # theta true
            loss_tmp,_ = f_max(model_lst, theta_true, n_dim)
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
        fig.savefig("sensitivity_analysis/SI_%s_n_exp=%s_n=%s_gap=%s_seed=%s.png"%(case_study,n_exp,N_steps,args.gap,data_seed))
        df_SI.to_csv("sensitivity_analysis/SI_%s_n_exp=%s_n=%s_gap=%s_seed=%s.csv"%(case_study,n_exp,N_steps,args.gap,data_seed))


if __name__ ==  '__main__':
    main()
