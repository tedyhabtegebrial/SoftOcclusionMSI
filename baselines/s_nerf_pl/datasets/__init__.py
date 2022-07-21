from .blender import BlenderDataset
from .llff import LLFFDataset
from .residential import ResidentialDataset
from .med_port import MedPortDataset
from .replica import ReplicaDataset
from .coffee_1d import CoffeArea1DDataset
from .coffee_2d import CoffeArea2DDataset
dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'coffee_1d': CoffeArea1DDataset,
                'coffee_2d': CoffeArea2DDataset,
                'residential': ResidentialDataset,
                'med_port': MedPortDataset,
                'replica': ReplicaDataset}
