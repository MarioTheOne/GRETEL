
# GRETEL (Graph counteRfactual ExplanaTion EvaLuation framework)


GRETEL is an open source framework for Evaluating Graph Counerfactual Explanation Methods. It is implemented using the Object Oriented paradigm and the Factory Method design pattern. Our main goal is to create a generic plataform that allows the researchers to speed up the proccess of developing and testing new Graph Counterfactual Explanation Methods.


## Requirements:

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


## Resources provided with the Framework:

### Datasets:

* **Tree-Cycles** [1]: Synthetic data set where each instance is a graph. The instance can be either a tree or a tree with several cycle patterns connected to the main graph by one edge

* **Tree-Infinity**: It follows the approach of the Tree-Cycles, but instead of cycles, there is an infinity shape.

* **ASD** [2]: Autism Spectrum Disorder (ASD) taken from the Autism Brain Imagine Data Exchange (ABIDE).

* **ADHD** [2]: Attention Deficit Hyperactivity Disorder (ADHD), is taken from the USC Multimodal Connectivity Database (USCD).

* **BBBP** [3]: Blood-Brain Barrier Permeation is a molecular dataset. Predicting if a molecule can permeate the blood-brain barrier.

* **HIV** [3]: It is a molecular dataset that classifies compounds based on their ability to inhibit HIV.


### Oracles:

* **KNN**

* **SVM**

* **GCN**

* **ASD Custom Oracle** [2] (Rules specific for the ASD dataset)


### Explainers:

* **DCE Search**: Distribution Compliant Explanation Search,  mainly used as a baseline, does not make any assumption about the underlying dataset and searches for a counterfactual instance in it.

* **Oblivious Bidirectional Search (OBS)** [2]: It is an heuristic explanation method that uses a 2-stage approach.

* **Data-Driven Bidirectional Search (DBS)** [2]: It follows the same logic as OBS. The main difference is that this method uses the probability (computed on the original dataset) of each edge to appear in a graph of a certain class to drive the counterfactual search process.

* **MACCS** [3]: Model Agnostic Counterfactual Compounds with STONED (MACCS) is specifically designed to work with molecules.


## How to use:

Lets see an small example of how to use the framework.

### Config file

First, we need to create a config json file with the option we want to use in our experiment. In the file config/CIKM/manager_config_example_all.json it is possible to find all options for each componnent of the framework.

![Example GRETEL config file](/examples/images/config_example.png "Example GRETEL config file")

Then to execute the experiment from the main the code would be something like this:

![Example GRETEL main](/examples/images/main_example.png "Example GRETEL main")

Once the result json files are generated it is possible to use the result_stats.py module to generate the tables with the results of the experiments. The tables will be generated as CSV and LaTex. In the examples folder there are some jupyter notebooks, and associated configuration files, that show how to use the framework for evaluating an explainer. Furthermore, they show how to extend GRETEL with new datasets and explainers.


## References

1. Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure Leskovec. 2019. Gnnexplainer: Generating explanations for graph neural networks. Ad-
vances in neural information processing systems 32 (2019)

2. Carlo Abrate and Francesco Bonchi. 2021. Counterfactual Graphs for Explainable Classification of Brain Networks. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2495–2504

3. Geemi P Wellawatte, Aditi Seshadri, and Andrew D White. 2022. Model agnostic generation of counterfactual explanations for molecules. Chemical science 13, 13
(2022), 3697–370