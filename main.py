import os
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=1 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=1

from src.evaluation.evaluator_manager import EvaluatorManager
import sys
from src.utils.logger import GLogger


# config_file_path = './config/steel/set-1/config_autism_custom-oracle_dce.json'

# print('Creating the evaluation manager.......................................................')
# eval_manager = EvaluatorManager(config_file_path, run_number=1)

# print('Creating the evaluators...................................................................')
# eval_manager.create_evaluators()

# print('Evaluating the explainers..................................................................')
# eval_manager.evaluate()


#GLogger._path="log" #Set the directory only once

logger = GLogger.getLogger()

#config_file_path = './linux-server/manager_config_lite.json'
# config_file_path = './config/linux-server/manager_caliban_lite.json'

config_file_path = sys.argv[1]
runno= int(sys.argv[2])

logger.info(f'Executing: {sys.argv[1]} Run: {runno}')

logger.info('Creating the evaluation manager.......................................................')
eval_manager = EvaluatorManager(config_file_path, run_number=runno)

# logger.info('Generating Synthetic Datasets...........................................................')
# eval_manager.generate_synthetic_datasets()

# logger.info('Training the oracles......................................................................')
# eval_manager.train_oracles()

logger.info('Creating the evaluators...................................................................')
eval_manager.create_evaluators()

logger.info('Evaluating the explainers..................................................................')
eval_manager.evaluate()
