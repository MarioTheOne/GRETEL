from typing import Set
from src.evaluation.evaluation_metric_ged import GraphEditDistanceMetric
from src.dataset.data_instance_base import DataInstance
from src.explainer.meg.environments.base_env import BaseEnvironment, Result
import copy
import numpy as np

class TreeCyclesEnvironment(BaseEnvironment):
    
    def __init__(self,
                 init_instance=None,
                 target_fn=None,
                 max_steps=10,
                 record_path=False):
        
        super().__init__(target_fn=target_fn,
                         max_steps=max_steps)
        self.init_instance = init_instance
        self._valid_actions = []
        self.record_path = record_path
        self._path = []
        self.reward_fn = GraphEditDistanceMetric().evaluate
        
    def get_path(self):
        return self._path

    def initialize(self):
        self._state = self.init_instance
        if self.record_path:
            self._path = [self._state]
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter = 0
        
    def get_valid_actions(self, state=None, force_rebuild=False):
        if state is None:
            if self._valid_actions and not force_rebuild:
                return copy.deepcopy(self._valid_actions)
            state = self._state
        self._valid_actions = self._get_valid_actions(state)
        return copy.deepcopy(self._valid_actions)
    
    def _get_valid_actions(self, state: DataInstance) -> Set[DataInstance]:
        adj_matrix = state.to_numpy_array()
        nodes = list(range(adj_matrix.shape[0]))
        valid_actions = []
        # Iterate through each node
        for node in nodes:
            # Iterate through neighbouring nodes and check for valid actions
            for neighbour in nodes:
                if neighbour > node:
                    # Adding/removal of edges
                    temp_inst = DataInstance(self.state._id + neighbour + 1)
                    adj_matrix[node][neighbour] = 1 - adj_matrix[node][neighbour]
                    adj_matrix[neighbour][node] = adj_matrix[node][neighbour]
                    temp_inst.from_numpy_array(adj_matrix)
                    valid_actions.append(temp_inst)

        return set(valid_actions)                
    
    def reward(self):     
        return {
            'reward': self.reward_fn(self._state, self.init_instance)
        }
    
    def step(self, action):
        if self.num_steps_taken >= self.max_steps or self.goal_reached():
            raise ValueError('This episode is terminated.')
        if action.id not in [inst.id for inst in self._valid_actions]:
            raise ValueError('Invalid action.')
        self._state = action
        if self.record_path:
            self._path.append(self._state)
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter += 1

        result = Result(
            state=self._state,
            reward=self.reward(),
            terminated=(self._counter >= self.max_steps) or self.goal_reached())
        
        return result