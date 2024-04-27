# BO4IO
This repository contains codes and data for the BO4IO paper. The following are the instructions to run the codes.

## Citation
Coming soon


## Requirements

Dependencies can be found in the `environment.yml` file.

We use Gurobi as the primary optimization solver. [here](https://www.gurobi.com/)

## Usage
To reproduce the results for each case study in the paper, clone the repository, install the required package and solver, and follow the instructions below.

### Steps to run the FBA case study

This case study aims to learn the cellular objective in the FBA problem. All codes should be executed under the `FBA` folder.

1. **Run BO4IO under specific problem settings**
   
   `python BO4IO_FBA.py -nexp 50 -nrxn 3 -ntrial 5 -niter 250 -ninit 5 -n_data_seed 1 -init_seed 1 -mp 1 -n_p 10 -sigma 0.1`

   Description of program inputs:

   * `nexp` - number of training experiments (FOPs)
   * `nrxn` - number of reactions (common cellular objectives) included in the multiobjective, i.e. dimensions of unknown parameters
   * `ntrial` - number of BO trials
   * `niter` - number of BO iterations
   * `ninit` - number of initial samplings for building GP
   * `n_data_seed` - number of random instances to be run in the program
   * `init_seed` - the starting seed of the random instance
   * `mp` - with (1) or without (0) multiprocessing/parallelization
   * `n_p` - number of parallelized workers
   * `sigma` - noise level on observed decisions
     
2. **Perform profile likelihood analysis**
   
   `python BO4IO_Post_PL.py -nexp 50 -nrxn 3 -ntrial 5 -niter 250 -ninit 5 -n_data_seed 1 -init_seed 1 -n_p 10 -sigma 0.1`

### Steps to run the standard pooling problem case study

This case study aims to learn the market demand in the standard pooling problem. All codes should be executed under the `standard_pooling` folder.

1. **Run BO4IO under specific problem settings**

   `python BO4IO_Std_Pooling.py -nexp 50 -niter 100 -case_study haverly1 -ntrial 5 -sigma 0.05 -theta_seed 1 -n_data_seed 1 -mp 1 -n_p 10 -ntheta 2`

   Additional program inputs:

   * `case_study` - name of benchmark problems, including `haverly1`, `foulds2`, and `foulds3`
   * `theta_seed` - the starting seed of the random instance, same as `init_seed`
   * `ntheta` - number of unknown parameters
     
2. **Perform profile likelihood analysis**

   `python BO4IO_PostPL.py -nexp 50 -niter 100 -case_study haverly1 -ntrial 5 -sigma 0.05 -theta_seed 1 -n_data_seed 1 -mp 1 -n_p 10 -ntheta 2`
   
3. **Perform sensitivity analysis**

   `python BO4IO_Std_Pooling_sensitivity_analysis.py -case_study haverly1  -ntheta 2 -nexp 50 -mp 1 -n_p 10 -theta_seed 1 -n_data_seed 1 -sigma 0.05`

   Additional program inputs:
   
   `nsteps` - number of equally-spaced points where the sensitivity analysis perform
   
### Steps to run the generalized pooling problem case study

This case study aims to learn the product quality requirements in the generalized pooling problem. All codes should be executed under the `standard_pooling` folder.

1. **Run BO4IO under specific problem settings**
   
   `python BO4IO_Gen_Pooling.py -nexp 50 -niter 200 -case_study Lee -ntrial 5 -sigma 0.05 -theta_seed 1 -n_data_seed 1 -mp 1 -n_p 10`
   
2. **Perform sensitivity analysis**
   
   `python BO4IO_Gen_Pooling_sensitivity_analysis.py -case_study Lee  -nexp 50 -mp 1 -n_p 10 -theta_seed 1 -n_data_seed 1 -sigma 0.05`
