from src.evaluation.evaluator_manager import EvaluatorManager
import sys


#config_file_path = 'C:\\Work\\GNN\\Mine\\Themis\\config\\windows-local\\manager_config_lite.json'
config_file_path = '/NFSHOME/mprado/CODE/Themis/config/CIKM/manager_config_example_all.json'

#config_file_path = sys.argv[1]
runno= 0

print('Creating the evaluation manager.......................................................')
eval_manager = EvaluatorManager(config_file_path, run_number=runno)

print('Generating Synthetic Datasets...........................................................')
eval_manager.generate_synthetic_datasets()

# print('Training the oracles......................................................................')
# eval_manager.train_oracles()

#print('Creating the evaluators...................................................................')
#eval_manager.create_evaluators()

#print('Evaluating the explainers..................................................................')
#eval_manager.evaluate()
