import torch
from torch import nn

from configs.synthetic import config
from src.data import ChainEquationModel
from src.models.irm import InvariantRiskMinimization
from src.utils import *


all_methods = {
        "IRM": InvariantRiskMinimization
    }


def run_experiment(config):
    if config["seed"] >= 0:
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        torch.set_num_threads(1)
    
    if config["setup_sem"] == "chain":
        setup_str = f'chain_ones={config["setup_ones"]}_hidden={config["setup_hidden"]}_hetero={config["setup_hetero"]}_scramble={config["setup_scramble"]}'
    


    if config["methods"] == "all":
        methods = all_methods
    else:
        methods = {m: all_methods[m] for m in config["methods"]}

    all_sems = []
    all_solutions = []
    all_envirionments = []

    for rep_i in range(config["n_reps"]):
        if config["setup_sem"] == "chain":
            sem = ChainEquationModel(
                dim=config["dim"],
                ones=config["setup_ones"],
                hidden=config["setup_hidden"],
                scramble=config["setup_scramble"],
                hetero=config["setup_hetero"]
            )
            env_list = config["env_list"]
            environments = [sem(config["n_samples"], e) for e in env_list]
        
    all_sems.append(sem)
    all_envirionments.append(environments)

    for sem, environment in zip(all_sems, all_envirionments):
        sem_solution, sem_scramble = sem.solution()

        solutions = [
            f'{setup_str} SEM{pretty(sem_solution)} {0:.5f} {0:.5f}'
        ]

        for method_name, method_constructor in methods.items():
            method = method_constructor(
                lr=config["lr"],
                iteration=config["n_iterations"],
                environments=environment,
                verbose=config["verbose"],
                reg=config["reg"]
            )

            method_solution = sem_scramble @ method.solution()
            
            err_causal, err_noncausal = weight_errors(sem_solution, method_solution)

            solutions.append(
                f"{setup_str} {method_name}{pretty(method_solution)} {err_causal:.5f} {err_noncausal:.5f}"
            )

        all_solutions += solutions

    return all_solutions

if __name__ == "__main__":
    all_solutions = run_experiment(config)
    print(all_solutions)