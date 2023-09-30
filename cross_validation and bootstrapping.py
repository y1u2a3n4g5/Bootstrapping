#%% Packages
import numpy as np
import pandas as pd
#%%  Data generating
# Create a sample matrix (replace this with your own data)
matrix = np.random.randint(1,51,size = (1000,5))
# Number of bootstrap samples to generate
num_samples = 100
#%%  Cross_validation
cross_validation_matrices = []

for i in range(10):
    # Create a mask for the rows to include in the training set
    train_mask = np.ones(len(matrix), dtype=bool)
    train_mask[100*i:100*(i+1)] = False  # Exclude a specific range of rows
    
    # Create a mask for the rows to include in the validation set
    val_mask = ~train_mask
    
    # Split the data into training and validation sets
    train_data = matrix[train_mask]
    val_data = matrix[val_mask]
    combined_matrix = np.vstack((train_data, val_data))
    # Append the training and validation sets to the list
    cross_validation_matrices.append(combined_matrix)
cross_validation_matrices = np.array(cross_validation_matrices)
#type(cross_validation_matrices)
#num_rows = len(cross_validation_matrices)
#num_columns = len(cross_validation_matrices[0])  
cross_validation_matrices.shape
#print(f"Number of rows: {num_rows}")
#print(f"Number of columns: {num_columns}")
#%% Bootstrapping
bootstrapped_matrices = []

for _ in range(num_samples):
    # Randomly sample rows from the matrix with replacement
    bootstrap_indices = np.random.choice(matrix.shape[0], size=matrix.shape[0], replace=True)
    bootstrap_matrix = matrix[bootstrap_indices, :]
    
    # Append the bootstrapped matrix to the list
    bootstrapped_matrices.append(bootstrap_matrix)
    
# Each element of bootstrapped_matrices is a bootstrapped matrix
# You can access these matrices as needed for further analysis
#%%
my_matrix = np.array(bootstrapped_matrices)
my_matrix.shape