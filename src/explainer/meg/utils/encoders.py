from abc import ABC, abstractmethod
import numpy as np
from src.dataset.data_instance_molecular import MolecularDataInstance
from src.dataset.data_instance_base import DataInstance
from src.explainer.meg.utils.fingerprints import Fingerprint
from rdkit.Chem import AllChem, MolFromSmiles
from rdkit import Chem


class ActionEncoderAB(ABC):
    
    def __init__(self):
        self._name = 'meg_action_encoder'
        
    @abstractmethod
    def encode(self, action: DataInstance) -> np.array:
        pass
    
    def encode_actions(self, actions):
        return [self.encode(action) for action in actions]
    
class TreeCyclesActionEncoder(ActionEncoderAB):
    
    def __init__(self):
        super(TreeCyclesActionEncoder, self).__init__()
        self._name = 'meg_tree_cycles_action_encoder'
        
    def encode(self, action: DataInstance) -> np.array:
        return action.to_numpy_array()
    
class MorganBitFingerprintActionEncoder(ActionEncoderAB):
    
    def __init__(self, fp_length=1024, fp_radius=2):
        super(MorganBitFingerprintActionEncoder, self).__init__()
        self._name = 'meg_morgan_bit_fingerprint_action_encoder'
        
        self.fp_length = fp_length
        self.fp_radius = fp_radius
        
    def encode(self, action: DataInstance) -> np.array:
        assert isinstance(action, MolecularDataInstance)
        
        action = MolFromSmiles(action.smiles())
        
        #if action: 
        fp = AllChem.GetMorganFingerprintAsBitVect(action,
                                                    self.fp_radius,
                                                    self.fp_length,
                                                    bitInfo=None)
        return Fingerprint(fp, self.fp_length).numpy()
        """else:
            raise ValueError(f'DataIntance with id={action.id}'\
                + f' does not have a valid SMILES representation')"""
    
class MorganCountFingerprintActionEncoder(ActionEncoderAB):
    
    def __init__(self, fp_length=1024, fp_radius=2):
        super(MorganCountFingerprintActionEncoder, self).__init__()
        self._name = 'meg_morgan_count_fingerprint_action_encoder'
        
        self.fp_length = fp_length
        self.fp_radius = fp_radius
        
    def encode(self, action: DataInstance) -> np.array:
        assert isinstance(action, MolecularDataInstance)
        
        action = MolFromSmiles(action.smiles())
        
        fp = AllChem.GetHashedMorganFingerprint(action,
                                                self.fp_radius,
                                                self.fp_length,
                                                bitInfo=None)
        return Fingerprint(fp, self.fp_length).numpy()
    
class RDKitFingerprintActionEncoder(ActionEncoderAB):
    
    def __init__(self, fp_length=1024, fp_radius=2):
        super(RDKitFingerprintActionEncoder, self).__init__()
        self._name = 'meg_rdkit_fingerprint_action_encoder'
        
        self.fp_length = fp_length
        self.fp_radius = fp_radius
        
    def encode(self, action: DataInstance) -> np.array:
        assert isinstance(action, MolecularDataInstance)
        
        action = MolFromSmiles(action.smiles())
        
        fp = Chem.RDKFingerprint(action,
                                 self.fp_radius,
                                 self.fp_length,
                                 bitInfo=None)
        return Fingerprint(fp, self.fp_length).numpy()
