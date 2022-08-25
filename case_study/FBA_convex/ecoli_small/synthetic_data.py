# import package
try:
    import cobra
except Exception as e:
    print(e)
import pandas as pd
import numpy as np
from pyomo.environ import *
import numpy as np
import numpy.random as npr
from pyDOE2 import *
import math
import multiprocessing as mp
# import multiprocess as mp
import time
import argparse
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

def model_reconstruction(model_name:str, ind = 1):
    """
    Read the SBML model to extract information (stoichiometric matrix, Reaction Set, Default Bounds, etc.) for Pyomo model
    """
    model = cobra.io.read_sbml_model(model_name)
    writer = pd.ExcelWriter('model_info.xlsx', engine='xlsxwriter')
    # get the stochiometric matrix
    s = cobra.util.array.create_stoichiometric_matrix(model, array_type = 'DataFrame')
    s.to_excel(writer, sheet_name='stochiometric matrix')
    # get the reaction and metabolite list
    RSet = s.columns.values.tolist()
    MetSet = s.index.values.tolist()
    # Determine the set size
    m = len(MetSet)
    n = len(RSet)
    # Get flux bound constraints 
    FB_con = cobra.util.array.constraint_matrices(model, array_type = 'DataFrame')[5]
    # Get default flux bounds
    UB_dict_WT = {}
    LB_dict_WT = {}
    for i in RSet:
        UB_dict_WT[i] = model.reactions.get_by_id(i).upper_bound
        LB_dict_WT[i] = model.reactions.get_by_id(i).lower_bound
    pd.DataFrame.from_dict([UB_dict_WT]).to_excel(writer, sheet_name='upper bound')
    pd.DataFrame.from_dict([LB_dict_WT]).to_excel(writer, sheet_name='lower bound')
    writer.save()    

    # Get stoichiometric matrix
    S = s.to_dict('index')
    S_dict = {}
    for i in MetSet:
        for j in RSet:
            S_dict[(i,j)] = S[i][j]
    if ind == 0:
        print("=========== Loading model name and sizes ===========")
        print("BiGG model ID: ", model_name)
        print("Number of species: ", m)
        print("Number of reactions: ", n)
        print(model.objective.expression, '\n')
    return RSet, MetSet, UB_dict_WT, LB_dict_WT, S_dict,

def read_model():
    UB_df = pd.read_excel('model_info.xlsx', sheet_name='upper bound',index_col = 0)
    UB_dict = UB_df.to_dict('records')[0]
    LB_df = pd.read_excel('model_info.xlsx', sheet_name='lower bound',index_col = 0)
    LB_dict = LB_df.to_dict('records')[0]
    s_df = pd.read_excel('model_info.xlsx', sheet_name='stochiometric matrix',index_col = 0)
    s_dict_tmp = s_df.to_dict('index')
    RSet = s_df.columns.values.tolist()
    MetSet = s_df.index.values.tolist()
    S_dict = {}
    for i in MetSet:
        for j in RSet:
            S_dict[(i,j)] = s_dict_tmp[i][j]
    return RSet, MetSet, UB_dict, LB_dict, S_dict

def get_redox_rxn_tuple(S_dict,RSet):
    """
    get the list of rxns that produce redox couple.
    return tuple of the rxn name and the stiohiometric coef of redox couples
    """
    redox_rxn_tuple = []
    redox_lst = ['nadh_c','nadph_c','fadh2_c']
    for i in redox_lst:
        for j in RSet:
            try:
                if S_dict[i,j] > 0:
                    redox_rxn_tuple.append((S_dict[i,j], j))
            except:
                pass
    return redox_rxn_tuple

def FBA(RSet, MetSet, UB_rand, LB_rand, S_dict, redox_rxn_tuple, c_ref, n_rxn = 3, scen = 1, ind = 1, solvername = 'gurobi'):
    """
    Create concreate Pyomo model for FBA
    """
    # count computation time
    start = time.time()
    # candidate reactions in objective functions (up to 5)
    obj_rxns_tup = [[(-1,'BIOMASS_Ecoli_core_w_GAM')], [(-1,'ATPM')], [(-1,'EX_glc__D_e')], [(-1,'EX_etoh_e')], redox_rxn_tuple]
    obj_rxns_lst = ['l2_norm','BIOMASS_Ecoli_core_w_GAM','ATPM','EX_glc__D_e','EX_etoh_e','REDOX POTENTIAL']
    model = ConcreteModel()
    model.r = Set(initialize=RSet)
    model.m = Set(initialize=MetSet)
    UB_dict = dict(zip(RSet,UB_rand))
    LB_dict = dict(zip(RSet,LB_rand))
    def Rb(model, i):
        return (LB_dict[i], UB_dict[i])
    model.R = Var(model.r, domain = Reals, bounds=Rb) 
    model.S = Param(model.m, model.r, initialize = S_dict) #takes time
    # variables and constrains for abs (l1) norm
    # model.U = Var(model.r, domain = NonNegativeReals)
    # def abs_cons_UB_rule(model,i):
    #     return model.R[i] <= model.U[i]
    # def abs_cons_LB_rule(model,i):
    #     return -model.R[i] <= model.U[i]
    # model.abscons_UB = Constraint(model.r, rule=abs_cons_UB_rule)
    # model.abscons_LB = Constraint(model.r, rule=abs_cons_LB_rule)
    # model.obj_FBA = Objective(expr=c_ref[0]*sum(model.U[i] for i in RSet)/len(RSet)+sum(c_ref[i]*obj_rxns_tup[i-1][j][0]*model.R[obj_rxns_tup[i-1][j][1]] for i in range(1,n_rxn) for j in range(len(obj_rxns_tup[i-1])) ), sense=minimize)
    model.obj_FBA = Objective(expr=c_ref[0]*sum(model.R[i]**2 for i in RSet)/len(RSet)+sum(c_ref[i]*obj_rxns_tup[i-1][j][0]*model.R[obj_rxns_tup[i-1][j][1]] for i in range(1,n_rxn) for j in range(len(obj_rxns_tup[i-1])) ), sense=minimize)
    def mb(model, i):
        return sum(model.S[(i,j)]*model.R[j] for j in RSet) == 0
    model.mbcons = Constraint(model.m, rule=mb) #takes time
    ti1 = time.time()-start
    # RSet_ex = ['EX_etoh_e','EX_for_e','EX_fru_e','EX_fum_e','EX_glc__D_e','EX_gln__L_e','EX_glu__L_e',
    # 'EX_lac__D_e','EX_pyr_e','EX_succ_e']
    # def substrate_limit(model):
    #     return sum(model.R[i] for i in RSet_ex) >= -1
    # model.subcons = Constraint(rule=substrate_limit)
    opt_FBA = SolverFactory(solvername)
    ti2 = time.time()-start - ti1
    try:
        results_FBA = opt_FBA.solve(model)#, tee=True)
        ts = time.time()-start-ti1 - ti2
        if results_FBA.solver.termination_condition == 'optimal':
            solutionList = []
            solutionList.append(0) # terminal status, 0: optimal, 1: other
            solutionList.append(value(model.obj_FBA))
            solutionList.append(sum(value(model.R[i])**2 for i in RSet)/len(RSet))
            for i in range(1,n_rxn):
                    solutionList.append(sum(value(model.R[obj_rxns_tup[i-1][j][1]]) for j in range(len(obj_rxns_tup[i-1]))))
            for k in RSet:
                solutionList.append(value(model.R[k]))
            if ind == 0:
                print("====================== FBA results =======================")
                # print("=========== Optimal fluxes of Wild-type cells ===========")
                # print(R_sol_WT)
                print("=========== Optimal Biomass fluxes ===========")
                print(value(model.obj_FBA), '\n')
                print("%s: %.3f \n"%('l2_norm', sum(value(model.R[i])**2 for i in RSet)/len(RSet)))
                for i in range(1,n_rxn):
                    print("%s: %.3f \n"%(obj_rxns_lst[i], sum(value(model.R[obj_rxns_tup[i-1][j][1]]) for j in range(len(obj_rxns_tup[i-1])))))
            
        else:
            solutionList = [1]
    except Exception as e: 
        print(e)
    tp = time.time()-start-ti1 - ti2-ts
    # ttol = time.time()-start
    return solutionList, UB_rand, LB_rand, ti1,ti2, ts, tp#, ttol

def generate_random_bounds(RSet, UB_dict_WT, LB_dict_WT, numSam = 10):
    # LB_dict_WT['BIOMASS_Ecoli_core_w_GAM'] = 0.005
    # Randomize flux bounds within 1%~100% WT range using Latin-Hypercube (lhs) sampling
    lhdnormc_UB = lhs(len(RSet), samples=numSam,random_state=2)  # Latin-Hypercube (lhs)
    lhdnormc_LB = lhs(len(RSet), samples=numSam,random_state=3)  # Latin-Hypercube (lhs)
    UB_rand = lhdnormc_UB  # linearly scale array lhdnorm from bound (0,1) to bound(1% UB_WT,100% UB_WT)
    LB_rand = lhdnormc_LB  # linearly scale array lhdnorm from bound (0,1) to bound(1% LB_WT,100% LB_WT)
    RSet_ex = ['EX_etoh_e','EX_for_e','EX_fru_e','EX_fum_e','EX_glc__D_e','EX_gln__L_e','EX_glu__L_e',
    'EX_lac__D_e','EX_pyr_e','EX_succ_e']


    for i in range(len(lhdnormc_UB[0, :])):
        if UB_dict_WT[RSet[i]] > 1000:
            UB_rand[:, i] = np.interp(lhdnormc_UB[:, i], (0, 1), (0.1, 100))
        else:
            UB_rand[:, i] = np.interp(lhdnormc_UB[:, i], (0, 1), (UB_dict_WT[RSet[i]]*0.0001, 0.1*UB_dict_WT[RSet[i]]))
        if LB_dict_WT[RSet[i]] < -1000:
            LB_rand[:, i] = np.interp(lhdnormc_LB[:, i], (0, 1), (-100,-0.1))
        elif any(RSet[i] in s for s in RSet_ex):
            LB_rand[:, i] = np.interp(lhdnormc_LB[:, i], (0, 1), (-10, 0))
        # elif RSet[i] == 'BIOMASS_Ecoli_core_w_GAM':
        #     LB_rand[:, i] = np.interp(lhdnormc_LB[:, i], (0, 1), (0.01, 0.01))
        else:
            LB_rand[:, i] = np.interp(lhdnormc_LB[:, i], (0, 1), (LB_dict_WT[RSet[i]]*0.1, LB_dict_WT[RSet[i]]*0.0001))



    return UB_rand, LB_rand

def list_to_chunks(sample_list, chunk_size):
    lst= lambda sample_list, chunk_size: [sample_list[i:i+chunk_size] for i in range(0, len(sample_list), chunk_size)]
    result=lst(sample_list, chunk_size)
    return result

def generate_feasible_set(RSet,MetSet, UB_dict_WT, LB_dict_WT, S_dict, redox_rxn_tuple,n_sample, c_Ref, n_pool, n_exp, scen = 1):
    # generate feasoble sets of ranmdom bounds for specified number of experiment
    UB_rand, LB_rand = generate_random_bounds(RSet, UB_dict_WT, LB_dict_WT, numSam = n_sample)
    # count computation time
    start = time.time()
    print('Start generating feasible sets:')
    # initialize df
    df_sol = pd.DataFrame()
    df_UB = pd.DataFrame()
    df_LB = pd.DataFrame()
    chunk_size = int(n_pool*10)
    UB_rand = list_to_chunks(UB_rand,chunk_size)
    LB_rand = list_to_chunks(LB_rand,chunk_size)
    # generate synthetic data
    count = 0
    n_rxn = 1
    c_Ref = [float(i)/sum(c_Ref[0:n_rxn]) for i in c_Ref]
    obj_rxns_lst = ['l2_norm','BIOMASS_Ecoli_core_w_GAM','ATPM','EX_glc__D_e','EX_etoh_e','REDOX POTENTIAL']

    try:
        while len(df_sol) < n_exp:
            # parallel computing (multiprocessing)
            p = mp.Pool(n_pool)
            batch_size = math.ceil(chunk_size/n_pool)
            solvername = 'ipopt'
            results = p.starmap(FBA, [(RSet, MetSet, UB_rand[count][i], LB_rand[count][i], S_dict, redox_rxn_tuple, c_Ref,n_rxn, scen,1,solvername) for i in range(chunk_size)],batch_size)
            p.close()
            p.join()

            # get solutions and bounds for fluxes of exps that are optimal
            solution = [r[0] for r in results]
            UB_rand_List = [r[1] for r in results]
            LB_rand_List = [r[2] for r in results]

            # store solutions and fesible bounds into df
            obj_rxns_lst = ['l2_norm','BIOMASS_Ecoli_core_w_GAM','ATPM','EX_glc__D_e','EX_etoh_e','REDOX POTENTIAL']
            column_name = ['status', 'obj_val'] + [obj_rxns_lst[i] for i in range(n_rxn)] + RSet 
            df_sol_tmp = pd.DataFrame(solution, columns = column_name)
            df_UB_tmp = pd.DataFrame(UB_rand_List, columns = RSet)
            df_LB_tmp = pd.DataFrame(LB_rand_List, columns = RSet)
            # filter out the infeasible exps
            df_sol_mock = df_sol_tmp
            df_sol_tmp = df_sol_tmp[df_sol_mock['status']==0]
            df_UB_tmp = df_UB_tmp[df_sol_mock['status']==0]
            df_LB_tmp = df_LB_tmp[df_sol_mock['status']==0]
            df_sol = pd.concat([df_sol, df_sol_tmp], ignore_index=True)
            df_UB = pd.concat([df_UB, df_UB_tmp], ignore_index=True)
            df_LB = pd.concat([df_LB, df_LB_tmp], ignore_index=True)
            count += 1
            print('iteration %d, number of feasible sets %d' %(count, len(df_sol)))

        # save solutions and bounds into one excel file
        writer = pd.ExcelWriter('synthetic_data_1.xlsx', engine='xlsxwriter')
        df_sol.to_excel(writer, sheet_name='flux solution')
        df_UB.to_excel(writer, sheet_name='upper bound')
        df_LB.to_excel(writer, sheet_name='lower bound')
        df_cref = pd.DataFrame(c_Ref[0:n_rxn])
        df_cref.to_excel(writer, sheet_name='C Ref')
        writer.save()
    except:
        print('Error: Not enough feasible sets, increase the value of the -f arguement.')
        writer = pd.ExcelWriter('synthetic_data_1.xlsx', engine='xlsxwriter')
        df_sol.to_excel(writer, sheet_name='flux solution')
        df_UB.to_excel(writer, sheet_name='upper bound')
        df_LB.to_excel(writer, sheet_name='lower bound')
        df_cref = pd.DataFrame(c_Ref[0:n_rxn])
        df_cref.to_excel(writer, sheet_name='C Ref')
        writer.save()
        # report computation time
    end = time.time()
    print('Feasible sets creation finished in %d sec.' %(end-start))
    
def main():
    """
    Python code for generating synthetic data with specified size of model
    Input arguements:
    1.  -nexp (--n_exp): number of exps
    2.  -p (--n_p): number of processors for multiprocessing
    3.  -s (--scen): linear (1) or nonlinear (2) objective
    4.  -f (--expand_f): expansion factor on numbers of random bounds. n_sample = n_exp*f
    5.  -ms (--mod_con): read sbml file to generate model info or not, 0 = yes, 1 = no

    Execute the code in terminal:
    python synthetic_data.py -nexp 1000 -p 8 -ms 1
    """
    # Collect input for model parameter assignment.
    parser = argparse.ArgumentParser(description='Runs parameter estimation to estimate Vmax in kinetic models (using exp data)')
    optional = parser._action_groups.pop()  # creates group of optional arguments
    required = parser.add_argument_group('required arguments')  # creates group of required arguments
    # required input
    
    # optional input
    optional.add_argument('-nexp', '--n_exp', help='number of exps', type=int, default = 1000)
    optional.add_argument('-p', '--n_p', help='number of processors for multiprocessing', type=int, default = 8)
    optional.add_argument('-s', '--scen', help='linear (1) or nonlinear (2) objective', type=int, default = 1)
    optional.add_argument('-f', '--expand_f', help='expansion factor on numbers of random bounds. n_sample = n_exp*f', type=int, default = 100)
    optional.add_argument('-ms', '--mod_con', help='read sbml file to generate model info or not, 0 = yes, 1 = no', type=int, default = 1)
    optional.add_argument('-fg', '--FSet_gen', help='Generate feasible sets of bounds or not, 0 = yes, 1 = no', type=int, default = 1)

    parser._action_groups.append(optional)  # add optional values to the parser
    args = parser.parse_args()  # get the arguments from the program input, set them to args
    
    # initialization
    n_exp = args.n_exp
    n_pool = args.n_p
    scen = args.scen
    expand_f = args.expand_f
    sbml_model = 'e_coli_core.xml'
    n_sample = int(n_exp*expand_f)

    if args.mod_con == 0:
        # extract model from sbml files
        RSet, MetSet, UB_dict_WT, LB_dict_WT, S_dict = model_reconstruction(sbml_model)
    else:
        RSet, MetSet, UB_dict_WT, LB_dict_WT, S_dict = read_model()
    # get the rxn list for the redox potential objective 
    redox_rxn_tuple = get_redox_rxn_tuple(S_dict,RSet)

    # generate reference cost vector
    np.random.seed(5) #10
    c_Ref = np.random.uniform(low=0.1, high=10, size=6).tolist()
    c_Ref[0] = 0.05

    # generate feasible sets or not
    if args.mod_con == 0:
        generate_feasible_set(RSet,MetSet, UB_dict_WT, LB_dict_WT, S_dict, redox_rxn_tuple,n_sample, c_Ref, n_pool, n_exp, scen = 1)
    
    # count computation time
    start = time.time()
    print('Data generator starts:')
    # define hypothesized objective list
    obj_rxns_lst = ['l2_norm','BIOMASS_Ecoli_core_w_GAM','ATPM','EX_glc__D_e','EX_etoh_e','REDOX POTENTIAL']
    # read feasible sets of bounds
    UB_exp_df = pd.read_excel('feasible_set.xlsx', sheet_name='upper bound',index_col = 0)
    LB_exp_df = pd.read_excel('feasible_set.xlsx', sheet_name='lower bound',index_col = 0)
    UB_rand = UB_exp_df[0:n_exp].values.tolist()
    LB_rand = LB_exp_df[0:n_exp].values.tolist()
    n_exp = len(UB_rand)
    print('feasible set: %d' %n_exp)
    for i in range(1,7):
        print('Generating data for objective with %d hypothesis' %i)
        n_rxn = i
        c_Ref = [float(i)/sum(c_Ref[0:n_rxn]) for i in c_Ref]
        # parallel computing (multiprocessing)
        p = mp.Pool(n_pool)
        batch_size = math.ceil(n_exp/n_pool)
        results = p.starmap(FBA, [(RSet, MetSet, UB_rand[i], LB_rand[i], S_dict, redox_rxn_tuple, c_Ref,n_rxn, scen) for i in range(n_exp)],batch_size)
        p.close()
        p.join()

        # get solutions and bounds for fluxes of exps that are optimal
        solution = [r[0] for r in results]
        UB_rand_List = [r[1] for r in results]
        LB_rand_List = [r[2] for r in results]

        # store solutions and fesible bounds into df
        column_name = ['status', 'obj_val'] + [obj_rxns_lst[i] for i in range(n_rxn)] + RSet 
        df_sol = pd.DataFrame(solution, columns = column_name)
        df_UB = pd.DataFrame(UB_rand_List, columns = RSet)
        df_LB = pd.DataFrame(LB_rand_List, columns = RSet)

        # save solutions and bounds into one excel file
        writer = pd.ExcelWriter('synthetic_data_%d.xlsx' %n_rxn, engine='xlsxwriter')
        df_sol.to_excel(writer, sheet_name='flux solution')
        df_UB.to_excel(writer, sheet_name='upper bound')
        df_LB.to_excel(writer, sheet_name='lower bound')
        df_cref = pd.DataFrame(c_Ref[0:n_rxn])
        df_cref.to_excel(writer, sheet_name='C Ref')
        writer.save()
    # report computation time
    end = time.time()
    print('Synthetic data finished in %d sec.' %(end-start))



if __name__ ==  '__main__':
    main()

    