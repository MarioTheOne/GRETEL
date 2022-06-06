# GRETEL (Graph counteRfactual ExplanaTion EvaLuation framework)


GRETEL is an open source framework for Evaluating Graph Counerfactual Explanation Methods. It is implemented using the Object Oriented paradigm and the Factory Method design pattern. Our main goal is to create a generic plataform that allows the researchers to speed up the proccess of developing and testing new Graph Counterfactual Explanation Methods.

## Requirements

* scikit-learn
* numpy 
* scipy
* pandas
* tensorflow (for GCN)
* jsonpickle (for serialization)
* joblib
* rdkit (Molecules)
* exmol (maccs method)
* networkx (Graphs)

## Example

Lets see an small example of how to use the framework.

### Config file

First, we need to create a config json file with the option we want to use in our experiment. In the file config/CIKM/manager_config_example_all.json it is possible to find all options for each componnent of the framework.

![Example GRETEL config file](/images/config_example.png "Example GRETEL config file")

Then to execute the experiment from the main the code would be something like this:

![Example GRETEL main](/images/main_example.png "Example GRETEL main")

Once the result json files are generate it is possible to use the result_stats.py module to generate the tables with the results of the experiments. The tables will be generated as CSV and LaTex