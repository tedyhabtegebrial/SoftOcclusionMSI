__all__ = ['get_dataset']
from .med_port import MedPortLoader
from .residential import ResidentialAreaLoader
from .replica import ReplicaLoader
from .nex_mpi import loadDataset
# from .dense_replica_loader import DenseReplicaLoader
# from .coffe_area_loader import CoffeAreaLoader
# from .coffee_area_1d import CoffeArea1DLoader
# from .coffee_area_2d import CoffeArea2DLoader
# from .coffee_area_12v import CoffeArea12VLoader
# from .coffee_area_r3 import CoffeAreaR3Loader
# from .coffee_area_r2 import CoffeAreaR2Loader
# from .coffee_area_r1 import CoffeAreaR1Loader

def get_dataset(configs):
    dataset_map = {}
    dataset_map['d3dkit'] = MedPortLoader
    dataset_map['residential'] = ResidentialAreaLoader
    dataset_map['replica'] = ReplicaLoader
    dataset_map['nex'] = loadDataset
    
    # dataset_map['coffee_1d'] = CoffeArea1DLoader
    # dataset_map['coffee_2d'] = CoffeArea2DLoader
    # dataset_map['coffee_12v'] = CoffeArea12VLoader
    # dataset_map['coffee_r3'] = CoffeAreaR3Loader
    # dataset_map['coffee_r2'] = CoffeAreaR2Loader
    # dataset_map['coffee_r1'] = CoffeAreaR1Loader
    # dataset_map['coffee'] = CoffeAreaLoader
    assert  configs.dataset in dataset_map.keys(), f'dataset mapping for {configs.dataset} not implemented'
    return dataset_map[configs.dataset](configs)
