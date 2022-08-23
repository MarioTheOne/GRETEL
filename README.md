
# GRETEL (Graph counteRfactual ExplanaTion EvaLuation framework)

## Table of Contents

* [General Information](#general-information)
* [Citation Request](#citation-request)
* [Requirements](#requirements)
* [Resources Provided with the Framework](#resources-provided-with-the-framework)
* [How to Use](#how-to-use)
* [References](#references)

## General Information:

GRETEL [1] is an open source framework for Evaluating Graph Counerfactual Explanation Methods. It is implemented using the Object Oriented paradigm and the Factory Method design pattern. Our main goal is to create a generic plataform that allows the researchers to speed up the proccess of developing and testing new Graph Counterfactual Explanation Methods.


## Citation Request:

Please cite our paper if you use GRETEL in your experiments:

Mario Alfonso Prado-Romero and Giovanni Stilo. 2022. GRETEL: Graph Counterfactual Explanation Evaluation Framework. In Proceedings of the 31st ACM International Conference on Information and Knowledge Management (CIKM '22). Association for Computing Machinery, New York, NY, USA. https://doi.org/10.1145/3511808.3557608

```
@inproceedings{prado-romero2022gretel,
  title={GRETEL: Graph Counterfactual Explanation Evaluation Framework},
  author={Prado-Romero, Mario Alfonso and Stilo, Giovanni},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  isbn = {9781450392365},
  year={2022},
  doi = {10.1145/3511808.3557608},
  booktitle={Proceedings of the 31st ACM International Conference on Information and Knowledge Management},
  location = {Atlanta, GA, USA},
  series = {CIKM '22}
}
```

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

* **Tree-Cycles** [2]: Synthetic data set where each instance is a graph. The instance can be either a tree or a tree with several cycle patterns connected to the main graph by one edge

* **Tree-Infinity**: It follows the approach of the Tree-Cycles, but instead of cycles, there is an infinity shape.

* **ASD** [3]: Autism Spectrum Disorder (ASD) taken from the Autism Brain Imagine Data Exchange (ABIDE).

* **ADHD** [3]: Attention Deficit Hyperactivity Disorder (ADHD), is taken from the USC Multimodal Connectivity Database (USCD).

* **BBBP** [4]: Blood-Brain Barrier Permeation is a molecular dataset. Predicting if a molecule can permeate the blood-brain barrier.

* **HIV** [4]: It is a molecular dataset that classifies compounds based on their ability to inhibit HIV.


### Oracles:

* **KNN**

* **SVM**

* **GCN**

* **ASD Custom Oracle** [3] (Rules specific for the ASD dataset)


### Explainers:

* **DCE Search**: Distribution Compliant Explanation Search,  mainly used as a baseline, does not make any assumption about the underlying dataset and searches for a counterfactual instance in it.

* **Oblivious Bidirectional Search (OBS)** [3]: It is an heuristic explanation method that uses a 2-stage approach.

* **Data-Driven Bidirectional Search (DBS)** [3]: It follows the same logic as OBS. The main difference is that this method uses the probability (computed on the original dataset) of each edge to appear in a graph of a certain class to drive the counterfactual search process.

* **MACCS** [4]: Model Agnostic Counterfactual Compounds with STONED (MACCS) is specifically designed to work with molecules.


## How to use:

Lets see an small example of how to use the framework.

### Config file

First, we need to create a config json file with the option we want to use in our experiment. In the file config/CIKM/manager_config_example_all.json it is possible to find all options for each componnent of the framework.

![Example GRETEL config file](/examples/images/config_example.png "Example GRETEL config file")

Then to execute the experiment from the main the code would be something like this:

![Example GRETEL main](/examples/images/main_example.png "Example GRETEL main")

Once the result json files are generated it is possible to use the result_stats.py module to generate the tables with the results of the experiments. The tables will be generated as CSV and LaTex. In the examples folder there are some jupyter notebooks, and associated configuration files, that show how to use the framework for evaluating an explainer. Furthermore, they show how to extend GRETEL with new datasets and explainers.


## References

1. Prado-Romero, Mario Alfonso, and Giovanni Stilo. "GRETEL: A unified framework for Graph Counterfactual Explanation Evaluation." arXiv preprint arXiv:2206.02957 (2022).

2. Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure Leskovec. 2019. Gnnexplainer: Generating explanations for graph neural networks. Ad-
vances in neural information processing systems 32 (2019)

3. Carlo Abrate and Francesco Bonchi. 2021. Counterfactual Graphs for Explainable Classification of Brain Networks. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2495–2504

4. Geemi P Wellawatte, Aditi Seshadri, and Andrew D White. 2022. Model agnostic generation of counterfactual explanations for molecules. Chemical science 13, 13
(2022), 3697–370