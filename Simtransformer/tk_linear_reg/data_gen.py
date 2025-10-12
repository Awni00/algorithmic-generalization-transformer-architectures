import torch
import torch.nn.functional as F
import math

class DatasetGenerator:
    def __init__(self, num_samples, seq_len, dim, sigma=0.0):
        """
        Initialize the dataset generator with the number of samples, sequence length, and dimension.
        
        :param num_samples: Number of samples to generate
        :param seq_len: Length of each sequence
        :param dim: Dimensionality of the input data x
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.dim = dim
        self.sigma = sigma
    
    def _generate_beta(self):
        """
        Generate a vector beta from a uniform distribution on the sphere with the specified dimension.
        
        :return: A beta vector of shape (dim, ) from the sphere
        """
        beta = torch.randn(self.dim)
        beta = F.normalize(beta, p=2, dim=0)  # Normalize to have unit length
        return beta
    
    def _generate_x(self):
        """
        Generate a matrix x where each row is sampled iid from the standard normal distribution.
        
        :return: A matrix x of shape (seq_len, dim) sampled from N(0,1)
        """
        return torch.randn(self.seq_len, self.dim)
    
    def _generate_y(self, x, beta):
        """
        Generate the target variable y as the dot product of x and beta.
        
        :param x: Input data matrix of shape (seq_len, dim)
        :param beta: Coefficient vector of shape (dim, )
        :return: Target variable y of shape (seq_len, )
        """
        return torch.matmul(x, beta) + torch.randn(self.seq_len) * math.sqrt(self.sigma)
      # Compute y = x · beta
    
    def generate_sample(self):
        """
        Generate a single sample consisting of x, y, and beta.
        
        :return: A dictionary with keys 'x', 'y', and 'beta'
        """
        x = self._generate_x() # shape (seq_len, dim)
        beta = self._generate_beta() # shape (dim, )
        y = self._generate_y(x, beta) # shape (seq_len, )
        return {'x': x, 'y': y, 'beta': beta}
    
    def generate_dataset(self):
        """
        Generate a dataset with the specified number of samples.
        
        :return: A list of samples, where each sample is a dictionary with keys 'x', 'y', and 'beta'
        """
        dataset = [self.generate_sample() for _ in range(self.num_samples)]
        return dataset
    
    def save_dataset(self, file_path):
        """
        Generate the dataset and save it to a file.
        
        :param file_path: The path to save the dataset
        """
        dataset = self.generate_dataset()
        torch.save(dataset, file_path)
        print(f"Dataset saved to {file_path}")

# Example usage
if __name__ == "__main__":
    import os
    # add parent directory to path
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from simtransformer.module_base import ConfigBase
    config = ConfigBase(config_dir=os.path.join(current_dir, 'configurations'))
    data_config = config.data_config
    data_dir = os.path.join(current_dir, data_config.data_dir)
    data_method = DatasetGenerator(
        num_samples=data_config.num_samples, 
        seq_len=data_config.seq_len, 
        dim=data_config.dim, 
        sigma=data_config.sigma
    )
    data_method.generate_dataset()
    data_method.save_dataset(os.path.join(data_dir, f"linear_d{data_config.dim}_L{data_config.seq_len}.pth"))
