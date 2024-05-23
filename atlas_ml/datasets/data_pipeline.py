from sklearn.datasets import load_diabetes

class DataPipeline:

    def __init__(self, configs:dict):
        
        self.configs = configs

    def get_dataset(self, dataset_name:str):

        if dataset_name == 'diabetes':
            return load_diabetes()
