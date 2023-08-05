from src.evaluation.evaluator_manager import EvaluatorManager
import sys

import wandb
import numpy as np 
import random


config_file_path = sys.argv[1]
runno = int(sys.argv[2])

print('Executing:'+sys.argv[1])

# config_file_path = './config/steel/cf2-tc28/config_tree-cycles-500-28_tc-custom-oracle_cf2_fold-0.json'
# runno = 1

# Define sweep config
sweep_configuration = {
    'method': 'bayes',
    'name': f'CF2_Runno={runno}',
    'metric': {'goal': 'maximize', 'name': 'Correctness'},
    'parameters': 
    {
        "alpha": {'values': list(np.arange(0.5, 1.01, 0.1, dtype=float))},
        "lam": {'values': [20, 100, 500, 1000]},
        "lr": {'values': [1e-4, 1e-3, 1e-2]},
        "epochs": {'values': [50,100,200,250,500]},
        "batch_size": {'values': [0.1, 0.15, 0.2]},
        "gamma": {'values': list(np.arange(0.1, 1.1, 0.1, dtype=float))},
     }
}

print('Creating the evaluation manager.......................................................')
eval_manager = EvaluatorManager(config_file_path, run_number=runno)

# Initialize sweep by passing in config. 
sweep_id = wandb.sweep(
  sweep=sweep_configuration, 
  project='GRETEL'
)

# print('Generating Synthetic Datasets...........................................................')
# eval_manager.generate_synthetic_datasets()

# print('Training the oracles......................................................................')
# eval_manager.train_oracles()

# sweep through the folds
def main():
    metric_reports = None

    for fold_id in range(1):
        run = wandb.init()
        # note that we define values from `wandb.config`  
        # instead of defining hard values
        alpha = wandb.config.alpha
        lam = wandb.config.lam
        epochs = wandb.config.epochs
        lr = wandb.config.lr
        batch_size = wandb.config.batch_size
        gamma = wandb.config.gamma
    
        print('Creating the evaluators...................................................................')
        eval_manager.create_evaluators()

        eval_manager.explainers[0].fold_id = fold_id
        eval_manager.explainers[0].alpha = alpha
        eval_manager.explainers[0].lam = lam
        eval_manager.explainers[0].epochs = epochs
        eval_manager.explainers[0].lr = lr
        eval_manager.explainers[0].batch_size = batch_size
        eval_manager.explainers[0].gamma = gamma
        
        print('Evaluating the explainers..................................................................')
        eval_manager.evaluate()
        
        if metric_reports is None:
            # The metrics are not available in the evaluator manager until we create the evaluators
            metric_reports = {f'{metric.name}': [] for metric in eval_manager.evaluation_metrics}

        for evaluator in eval_manager.evaluators:
            for metric in eval_manager.evaluation_metrics:
                metric_reports[f'{metric.name}'].append(evaluator._results[f'{metric.name}'])

    wandb.log({
        f'{metric.name}': np.mean(metric_reports[f'{metric.name}']) for metric in eval_manager.evaluation_metrics
    })

# Start the sweep job
wandb.agent(sweep_id, function=main, count=20)
