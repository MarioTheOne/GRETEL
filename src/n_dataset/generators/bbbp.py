from os.path import join

import networkx as nx
import pandas as pd
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi

from src.n_dataset.generators.base import Generator
from src.n_dataset.instances.graph import GraphInstance


class BBBP(Generator):
    
    def init(self):
        base_path = self.local_config['parameters']['data_dir']
        self._bbbp_file_path = join(base_path, 'BBBP.csv')
        self.generate_dataset()
        
    def get_num_instances(self):
        return len(self.dataset.instances)
    
    def generate_dataset(self):
        if not len(self.dataset.instances):
            self.read(self._bbbp_file_path)
        
    def read(self, path):
        bbbp_data = pd.read_csv(path)
        bbbp_data = bbbp_data.sample(frac=1.0).reset_index(drop=True)

        skipped_molecules = 0
        for i in range(len(bbbp_data)):
            mol, smi, sanitized = self._sanitize_smiles(bbbp_data.smiles[i])
            if sanitized:
                self.dataset.instances.append(GraphInstance(id=i - skipped_molecules,
                                                            data=nx.to_numpy_array(self.mol_to_nx(mol)),
                                                            label=int(bbbp_data.p_np[i]),
                                                            graph_features=smi))
            else:
                skipped_molecules += 1                
                
    def _sanitize_smiles(self, smiles):
        try:
            mol = smi2mol(smiles, sanitize=True)
            smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
            check = smi2mol(smi_canon)
            # if the conversion failed
            if check is None:
                return None, None, False
            return mol, smi_canon, True
        except Exception as e:
            return None, None, False
        
    def mol_to_nx(self, mol):
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                    atomic_num=atom.GetAtomicNum(),
                    formal_charge=atom.GetFormalCharge(),
                    chiral_tag=atom.GetChiralTag(),
                    hybridization=atom.GetHybridization(),
                    num_explicit_hs=atom.GetNumExplicitHs(),
                    is_aromatic=atom.GetIsAromatic())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    bond_type=bond.GetBondType())
        return G