import os

"""os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=1 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=1"""

from src.evaluation.evaluator_manager import EvaluatorManager
from src.utils.context import Context
import sys

if __name__ == "__main__":
    print(f"Generating context for: {sys.argv[1]} Run: {sys.argv[2]}")
    context = Context.get_context(sys.argv[1])
    context.run_number = int(sys.argv[2])

    context.logger.info(f"Executing: {context.config_file} Run: {context.run_number}")
    context.logger.info(
        "Creating the evaluation manager......................................................."
    )

    context.logger.info(
        "Creating the evaluators..............................................................."
    )
    eval_manager = EvaluatorManager(context)

    context.logger.info(
        "Evaluating the explainers............................................................."
    )
    eval_manager.evaluate()
