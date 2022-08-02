from synthetic_data import *
def get_exp_input(n_exp,n_rxn):
    UB_exp_df = pd.read_excel('synthetic_data_%d.xlsx' %n_rxn, sheet_name='upper bound',index_col = 0)
    LB_exp_df = pd.read_excel('synthetic_data_%d.xlsx' %n_rxn, sheet_name='lower bound',index_col = 0)
    UB_exp = UB_exp_df[0:n_exp].values.tolist()
    LB_exp = LB_exp_df[0:n_exp].values.tolist()
    sol_exp_df = pd.read_excel('synthetic_data_%d.xlsx' %n_rxn, sheet_name='flux solution',index_col = 0)
    sol_exp_df = sol_exp_df.iloc[0:n_exp,n_rxn+2:]
    C_Ref_df = pd.read_excel('synthetic_data_%d.xlsx' %n_rxn, sheet_name='C Ref',index_col = 0)
    C_Ref = C_Ref_df.values.tolist()
    C_Ref = [item for sublist in C_Ref for item in sublist]
    return UB_exp, LB_exp, sol_exp_df, C_Ref
def simulator(c_vector, n_exp,n_rxn, n_pool):
    UB_exp, LB_exp, sol_exp_df, C_Ref = get_exp_input(n_exp,n_rxn)
    RSet, MetSet, UB_dict_WT, LB_dict_WT, S_dict = read_model()
    # get the rxn list for the redox potential objective 
    redox_rxn_tuple = get_redox_rxn_tuple(S_dict,RSet)
    p = mp.Pool(n_pool)
    batch_size = math.ceil(n_exp/n_pool)
    scen = 1
    results = p.starmap(FBA, [(RSet, MetSet, UB_exp[i], LB_exp[i], S_dict, redox_rxn_tuple, c_vector, n_rxn, scen) for i in range(n_exp)],batch_size)
    p.close()
    p.join()

    # get solutions and bounds for fluxes of exps that are optimal
    solution = [r[0] for r in results]
    # store solutions and fesible bounds into df
    obj_rxns_lst = ['abs_norm','BIOMASS_Ecoli','ATPM','EX_glc__D_e','EX_etoh_e','REDOX POTENTIAL']
    column_name = ['status', 'obj_val'] + [obj_rxns_lst[i] for i in range(n_rxn)] + RSet
    df_sol = pd.DataFrame(solution, columns = column_name)
    df_sol = df_sol.iloc[:,n_rxn+2:]
    sol_exp_ref = sol_exp_df.to_numpy()
    sol_exp = df_sol.to_numpy()
    loss = (sol_exp_ref - sol_exp)
    loss = np.multiply(loss,loss).sum()/n_exp
    return loss
def main():
    """
    Python code for simulator
    Input current cost vector, return current cost
    Input arguements:
    1.  -nexp (--n_exp): number of exps
    2.  -nrxn (--n_rnx): number of rxns in the objective
    3.  -c (--c_vec): current cost vector. Length should be same with -nrxn
    4.  -p (--n_p): number of processors for multiprocessing
    5.  -s (--scen): linear (1) or nonlinear (2) objective



    Execute the code in terminal:
    python simulator.py -nexp 10 -nrxn 3 -c 1 1 1
    """
    # Collect input for model parameter assignment.
    parser = argparse.ArgumentParser(description='Runs parameter estimation to estimate Vmax in kinetic models (using exp data)')
    optional = parser._action_groups.pop()  # creates group of optional arguments
    required = parser.add_argument_group('required arguments')  # creates group of required arguments
    # required input
    required.add_argument('-nexp', '--n_exp', help='number of exps', type=int, required=True)
    required.add_argument('-nrxn', '--n_rxn', help='number of rxns in the objective', type=int, required=True)
    required.add_argument('-c','--c_vec', nargs='+', help='current cost vector. Length should be same with -nrxn', required=True)
    # optional input
    optional.add_argument('-p', '--n_p', help='number of processors for multiprocessing', type=int, default = 8)
    optional.add_argument('-s', '--scen', help='linear (1) or nonlinear (2) objective', type=int, default = 1)

    parser._action_groups.append(optional)  # add optional values to the parser
    args = parser.parse_args()  # get the arguments from the program input, set them to args
    
    c_vector = [float(i) for i in args.c_vec]
    n_exp = args.n_exp
    n_rxn = args.n_rxn
    n_pool = args.n_p

    if len(c_vector) != n_rxn:
        print('Error: length of c vector is not equal to -nrxn!')

    print('Simulator starts:')
    # count computation time
    start = time.time()
    loss = simulator(c_vector, n_exp,n_rxn, n_pool)
    with open('loss.txt', 'w') as f:
        f.write('current loss: %.10f' %loss)
    end = time.time()
    print('Simulator finished in %d sec.' %(end-start))

    print('current loss: %.10f' %loss)

    
if __name__ ==  '__main__':
    main()
