from src.evaluation.evaluator_manager import EvaluatorManager
import sys

import wandb
import numpy as np 
import random


# config_file_path = sys.argv[1]
# runno = int(sys.argv[2])

# print('Executing:'+sys.argv[1])

config_file_path = './config/steel/clear-tc28/config_tree-cycles-500-28_tc-custom-oracle_clear_fold-0.json'
runno = 1

# Define sweep config
sweep_configuration = {
    'method': 'grid',
    'name': f'Runno={runno}',
    'metric': {'goal': 'maximize', 'name': 'Correctness'},
    'parameters': 
    {

            "batch_size_ratio": {'values': list(np.arange(0.1, 0.3, 0.1, dtype=float))},
            "alpha": {'values': list(np.arange(0.1, 0.8, 0.1, dtype=float))},
            "lr": {'values': list(np.arange(0.001, 0.01, 0.001, dtype=float))},
            "epochs": {'values': list(range(10, 200, 10))},
            "dropout": {'values': list(np.arange(0.01, 0.1, 0.01, dtype=float))},
            "weight_decay": {'values': [0.00001, 0.00004, 0.00007, 0.0001]},
            # fixed ones
            "graph_pool_type": {'values': ['mean']},
            "encoder_type": {'values': ['gcn']},
            "feature_dim": {'values': [2]},
            "h_dim": {'values': [16]},
            "z_dim": {'values': [16]},
            "disable_u": {'values': [False]},
            "beta_x": {'values': [10]},
            "beta_adj": {'values': [10]},
            "lambda_sim": {'values': [1]},
            "lambda_kl": {'values': [1]},
            "lambda_cfe": {'values': [1]}
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
        batch_size_ratio = wandb.config.batch_size_ratio
        alpha = wandb.config.alpha
        lr = wandb.config.lr
        epochs = wandb.config.epochs
        dropout = wandb.config.dropout
        weight_decay = wandb.config.weight_decay
        # fixed ones
        graph_pool_type = wandb.config.graph_pool_type
        encoder_type = wandb.config.encoder_type
        feature_dim = wandb.config.feature_dim
        h_dim = wandb.config.h_dim    
        z_dim = wandb.config.z_dim
        disable_u = wandb.config.disable_u    
        beta_x = wandb.config.beta_x
        beta_adj = wandb.config.beta_adj
        lambda_sim = wandb.config.lambda_sim
        lambda_kl = wandb.config.lambda_kl
        lambda_cfe = wandb.config.lambda_cfe
    
        print('Creating the evaluators...................................................................')
        eval_manager.create_evaluators()

        eval_manager.explainers[0].fold_id = fold_id

        eval_manager.explainers[0].batch_size_ratio = batch_size_ratio
        eval_manager.explainers[0].alpha = alpha
        eval_manager.explainers[0].lr = lr
        eval_manager.explainers[0].epochs = epochs
        eval_manager.explainers[0].dropout = dropout
        eval_manager.explainers[0].weight_decay = weight_decay
        
        eval_manager.explainers[0].graph_pool_type = graph_pool_type
        eval_manager.explainers[0].encoder_type = encoder_type
        eval_manager.explainers[0].feature_dim = feature_dim
        eval_manager.explainers[0].h_dim = h_dim
        eval_manager.explainers[0].z_dim = z_dim
        eval_manager.explainers[0].disable_u = disable_u
        eval_manager.explainers[0].beta_x = beta_x
        eval_manager.explainers[0].beta_adj = beta_adj
        eval_manager.explainers[0].lambda_sim = lambda_sim
        eval_manager.explainers[0].lambda_kl = lambda_kl
        eval_manager.explainers[0].lambda_cfe = lambda_cfe
        
        print('Evaluating the explainers..................................................................')
        eval_manager.evaluate()
        
        if metric_reports is None:
            # The metrics are not available in the evaluator manager until we create the evaluators
            metric_reports = {f'{metric.name}': [] for metric in eval_manager.evaluation_metrics}

        for evaluator in eval_manager.evaluators:
            for metric in eval_manager.evaluation_metrics:
                metric_reports[f'{metric.name}'].append(evaluator._results[f'{metric.name}'])

    # metrics_to_log = {}
    # for metric in eval_manager.evaluation_metrics:
    #     metrics_to_log[f'{metric.name}'] = np.mean(metric_reports[f'{metric.name}'])

    # b_metrics = [metric_reports[f'{metric.name}'] for metric in eval_manager.evaluation_metrics]

    wandb.log({
        f'{metric.name}': np.mean(metric_reports[f'{metric.name}']) for metric in eval_manager.evaluation_metrics
    })

# Start the sweep job
wandb.agent(sweep_id, function=main, count=20)
