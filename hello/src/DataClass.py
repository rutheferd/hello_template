from dataclasses import dataclass

from tensorflow import keras


class Singleton(type):
    _instances = {}

    def __call__(cls, *args , **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

@dataclass(frozen=False)
class Parameters(metaclass=Singleton):
    num_epochs: int = 5
    batch_size: int = 10
    height_pixels: int = 400
    width_pixels: int = 400
    training_file: str = ""
    test_file: str = ""
    model_file: str = ""
    num_confidence: int = -1
    output_location: str = ""






#data = Parameters(2,10,400,400, training_file="Grapevine_Leaves_Image_Dataset", test_file="testing_set\\Grapevine",
#                            model_file= "CDSavedModel" )