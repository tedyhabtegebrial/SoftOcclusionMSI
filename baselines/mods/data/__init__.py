__all__ = ['get_dataset']
from .residential_loader import ResidentialLoader
from .med_port import MedPortLoader
from .replica_loader import ReplicaLoader

def get_dataset(configs):
    dataset_map = {}
    dataset_map['residential'] = ResidentialLoader
    dataset_map['med_port'] = MedPortLoader
    dataset_map['replica'] = ReplicaLoader
    assert  configs.dataset in dataset_map.keys(), f'dataset mapping for {configs.dataset} not implemented'
    return dataset_map[configs.dataset](configs)
