from rdkit import Chem
from rdkit.Chem import Draw


mol = Chem.MolFromSmiles('Cc1ccccc1')
img = Draw.MolToImage(m)

for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                    atomic_num=atom.GetAtomicNum(),
                    formal_charge=atom.GetFormalCharge(),
                    chiral_tag=atom.GetChiralTag(),
                    hybridization=atom.GetHybridization(),
                    num_explicit_hs=atom.GetNumExplicitHs(),
                    is_aromatic=atom.GetIsAromatic())

