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
    'name': f'Countergan_Runno={runno}',
    'metric': {'goal': 'maximize', 'name': 'Correctness'},
    'parameters': 
    {
        "batch_size_ratio": {'values': [0.1, 0.15, 0.2]},
        'training_iterations': {'values': list(range(10, 101, 10))},
        'n_generator_steps': {'values': [10, 30, 100]},
        "n_discriminator_steps": {'values': [10, 30, 100]},
        "ce_binarization_threshold": {'values': [0.4, 0.5, 0.6, 0.7, 0.8]}
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

    for fold_id in range(10):
        run = wandb.init()
        # note that we define values from `wandb.config`  
        # instead of defining hard values
        batch_size_ratio = wandb.config.batch_size_ratio
        training_iterations = wandb.config.training_iterations
        n_generator_steps = wandb.config.n_generator_steps
        n_discriminator_steps = wandb.config.n_discriminator_steps
        ce_binarization_threshold = wandb.config.ce_binarization_threshold
    
        print('Creating the evaluators...................................................................')
        eval_manager.create_evaluators()

        eval_manager.explainers[0].fold_id = fold_id
        eval_manager.explainers[0].batch_size_ratio = batch_size_ratio
        eval_manager.explainers[0].training_iterations = training_iterations
        eval_manager.explainers[0].n_generator_steps = n_generator_steps
        eval_manager.explainers[0].n_discriminator_steps = n_discriminator_steps
        eval_manager.explainers[0].ce_binarization_threshold = ce_binarization_threshold
        
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
wandb.agent(sweep_id, function=main, count=3)
