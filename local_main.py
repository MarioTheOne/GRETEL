
import sys

from src.evaluation.evaluator_manager import EvaluatorManager

print(f"Initializing test ensemble")

# config_file_path = './config/steel/meg-set-1/config_tree-cycles-500-32_tc-custom-oracle_meg_fold-0.json'
config_file_path = './config/steel/cf2-bbbp/config_bbbp_gcn-tf_cf2_fold-0'

print('Creating the evaluation manager.......................................................')
eval_manager = EvaluatorManager(config_file_path, run_number=0)
# print('Generating Synthetic Datasets...........................................................')
# eval_manager.generate_synthetic_datasets()
#print('Training the oracles......................................................................')
#eval_manager.train_oracles()
print('Creating the evaluators...................................................................')
eval_manager.create_evaluators()
print('Evaluating the explainers..................................................................')
eval_manager.evaluate()


# for ds in ["1", "4", "5"]:
#     print(f"Initializing test syn{ds}")

#     config_file_path = f'./config/linux-server/set-1/config_syn{ds}_gcn-synthetic-pt_cfgnnexplainer.json'

#     print('Creating the evaluation manager.......................................................')
#     eval_manager = EvaluatorManager(config_file_path, run_number=0)

#     # print('Generating Synthetic Datasets...........................................................')
#     # eval_manager.generate_synthetic_datasets()

#     # print('Training the oracles......................................................................')
#     # eval_manager.train_oracles() 

#     print('Creating the evaluators...................................................................')
#     eval_manager.create_evaluators()

#     print('Evaluating the explainers..................................................................')
#     eval_manager.evaluate()

# <BEGIN> Generating stats tables ////////////////////////////////////////////////////////////////////////////////////////////////////
# data_store_path = "C:\\Work\\GNN\\Mine\\Themis\\data\\datasets\\"
# dtan = DataAnalyzer('C:\\Work\\GNN\\Mine\\Themis\\output', 'C:\\Work\\GNN\\Mine\\Themis\\output')

# dtan.aggregate_data()
# dtan.aggregate_runs()
# dtan.create_tables_by_oracle_dataset()

# datasets =[
#     {"name": "adhd", "parameters": {} },
#     {"name": "autism", "parameters": {} },
#     {"name": "tree-cycles", "parameters": {"n_inst": 500, "n_per_inst": 300, "n_in_cycles": 200} },
#     {"name": "tree-infinity", "parameters": {"n_inst": 500, "n_per_inst": 300, "n_infinities": 10, "n_broken_infinities": 10}},
#     {"name": "bbbp", "parameters": {"force_fixed_nodes": False}},
#     {"name": "hiv", "parameters": {"force_fixed_nodes": False}}
# ]

# dtan.get_datasets_stats(datasets, data_store_path)

# <END> Generating stats tables ////////////////////////////////////////////////////////////////////////////////////////////////////

# sl = SmilesLevenshteinMetric()

# mdi1 = MolecularDataInstance(1)
# mdi1.smiles = 'C(=O)(OC(C)(C)C)CCCc1ccc(cc1)N(CCCl)CCCl'
# mdi2 = MolecularDataInstance(2)
# mdi2.smiles = 'C(=O)(OC(C)(C)C)CCCc1ccc(cc1)N(CCCl)CNClCC'

# print(sl.evaluate(mdi1, mdi2))

# dsf = DatasetFactory('C:\\Work\\GNN\\Mine\\Themis\\data\\datasets\\')
# bbbp = dsf.get_bbbp_dataset(False, True, config_dict={})

# gcn_clf = TfGCNOracle(1, "C:\\Work\\GNN\\Mine\\Themis\\data\\oracles\\", config_dict={})

# gcn_clf.fit(bbbp, -1)

# for inst in bbbp.instances:
#     pred = gcn_clf.predict(inst)
#     print('original label:', inst.graph_label, 'predicted label:', pred)

# def clf_wrapper(name):
#     def dummy_classifier(smiles_mol: str) -> float:
#         print(name)
#         c_count = smiles_mol.count('c')
#         cc_count = smiles_mol.count('C')

#         if (c_count + cc_count) > 15:
#             return 0
#         else:
#             return 1

#     return dummy_classifier




# mol of interest
# base = 'CCC=OC'

# samples = exmol.sample_space(base, clf_wrapper('the horror'), batched=False, use_selfies=True)

# cfs = exmol.cf_explain(samples)
# exmol.plot_cf(cfs)

# print('finished')

# bbbp_data = BBBPDataset(0, None, True)
# bbbp_data.read_molecules_file('C:\\Work\\GNN\\Mine\\Themis\\data\\datasets\\bbbp\\')

# print('number of molecules readed:', len(bbbp_data.instances))

# for inst in bbbp_data.instances:
#     print('id:', inst.id)
#     print('name:', inst.name)
#     print(len(inst.graph.nodes))
#     print(len(inst.graph.edges))


# # Printing the molecule properties
# print('Original atoms.....................................')
# for at_0 in bbbp_data.instances[0].molecule.GetAtoms():
#     print(at_0.GetIdx(),',',
#           at_0.GetAtomicNum(),',',
#           at_0.GetIsAromatic(),',',
#           at_0.GetSymbol())

# print('Original bonds.....................................')
# for bnd_0 in bbbp_data.instances[0].molecule.GetBonds():
#     print(bnd_0.GetBeginAtomIdx(),',',
#           bnd_0.GetEndAtomIdx(),',',
#           bnd_0.GetBondType())

    
# bbbp_data.instances[0].molecule = None

# bbbp_data.instances[0].graph_to_molecule(True, True, max_n_atoms=bbbp_data.number_of_different_atoms, max_mol_len=bbbp_data.max_molecule_len)
# print('Transformed back ////////////////////////////////////////////////////')

# print('recovered atoms.....................................')
# # Printing the molecule properties
# for at_0 in bbbp_data.instances[0].molecule.GetAtoms():
#     print(at_0.GetIdx(),',',
#           at_0.GetAtomicNum(),',',
#           at_0.GetIsAromatic(),',',
#           at_0.GetSymbol())

# print('recovered bonds.....................................')
# for bnd_0 in bbbp_data.instances[0].molecule.GetBonds():
#     print(bnd_0.GetBeginAtomIdx(),',',
#           bnd_0.GetEndAtomIdx(),',',
#           bnd_0.GetBondType())

#     for node in inst.graph.nodes(data=True):
#         print(node)

# Loading the molecule
# caffeine_smiles = 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
# caffeine_mol = Chem.MolFromSmiles(caffeine_smiles)

# Printing the molecule properties
# for at_0 in caffeine_mol.GetAtoms():
#     print(at_0.GetIdx(),',',
#           at_0.GetAtomicNum(),',',
#           at_0.GetIsAromatic(),',',
#           at_0.GetSymbol())

# for bnd_0 in caffeine_mol.GetBonds():
#     print(bnd_0.GetBeginAtomIdx(),',',
#           bnd_0.GetEndAtomIdx(),',',
#           bnd_0.GetBondType())


# x = MolecularDataInstance()
# x.molecule_to_graph(caffeine_mol, False)

# Printing the graph nodes
# for ng in x.graph.nodes(data=True):
#     print(ng)

# for eg in x.graph.edges(data=True):
#     print(eg)

# coffee_molecule = x.graph_to_molecule()

# # Printing the recovered molecule properties
# for at_r in coffee_molecule.GetAtoms():
#     print(at_r.GetIdx(),',',
#           at_r.GetAtomicNum(),',',
#           at_r.GetIsAromatic(),',',
#           at_r.GetSymbol())

# for bnd_r in coffee_molecule.GetBonds():
#     print(bnd_r.GetBeginAtomIdx(),',',
#           bnd_r.GetEndAtomIdx(),',',
#           bnd_r.GetBondType())