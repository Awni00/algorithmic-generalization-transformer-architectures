
# %% training
import os
from tk_linear_reg.custom import Config, Pipeline, DataModule, TrainingManager
from simtransformer.module_base import DirectoryHandler
import argparse
import torch
import numpy as np
import random
from simtransformer.module_base import ConfigBase
from tk_linear_reg.data_gen import DatasetGenerator

current_dir = os.path.dirname(os.path.abspath(__file__))
task_dir = os.path.join(current_dir, 'tk_linear_reg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--only_dst', action='store_true', help='Whether to only use the destination entity.')

    args = parser.parse_args()
    
    
    



    config = ConfigBase(config_dir=os.path.join(task_dir, 'configurations'))
    data_config = config.data_config
    data_path = os.path.join(task_dir, "data", f"linear_d{data_config.dim}_L{data_config.seq_len}.pth")
    data_method = DatasetGenerator(
        num_samples=data_config.num_samples, 
        seq_len=data_config.seq_len, 
        dim=data_config.dim, 
        sigma=data_config.sigma
    )
    data_method.generate_dataset()
    data_method.save_dataset(data_path)


    dir_handler = DirectoryHandler(
        load_data_abs_dir=os.path.join(task_dir, 'data'),
        data_file_name=data_path,
        vocab_file_name='vocab.yaml', # it is not used
        load_config_abs_dir=os.path.join(task_dir, 'configurations'),
        load_ckpt_abs_path=None,
        output_abs_dir=None,
        create_run_under_abs_dir=task_dir,
        training_name=None,
    )
    training_manager = TrainingManager(
        dir_handler=dir_handler,
        use_wandb=True,
        abstract_config=Config, 
        abstract_pipeline=Pipeline,
        abstract_datamodule=DataModule,
    )
        
    training_manager.fit()


# # %% Continue training
# import os
# from tk_comp_atomic.custom import Config, Pipeline, DataModule, TrainingManager
# from simtransformer.module_base import DirectoryHandler
# current_dir = os.path.dirname(os.path.abspath(__file__))
# task_dir = os.path.join(current_dir, 'tk_comp_atomic')
# run_dir = os.path.join(task_dir, 'output', 'atomic_L2H4W0.0T0720')

# dir_handler = DirectoryHandler(
#     load_data_abs_dir=os.path.join(task_dir, 'data'),
#     data_file_name='atomic.json',
#     vocab_file_name='vocab.yaml',
#     load_config_abs_dir=os.path.join(run_dir, 'configurations'),
#     load_ckpt_abs_path=os.path.join(run_dir, 'best.ckpt'),
#     output_abs_dir=None,
#     create_run_under_abs_dir=task_dir,
# )

# training_manager = TrainingManager(
#     dir_handler=dir_handler,
#     use_wandb=False,
#     abstract_config=Config, 
#     abstract_pipeline=Pipeline,
#     abstract_datamodule=DataModule,
# )
# training_manager.fit()