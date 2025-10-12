from typing import Any, Optional, final, Union

from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
import lightning
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from .utils import CosineAnnealingWarmup, EasyDict, clever_load, clever_save, Shampoo, signSGD, normSGD, NormalizeSGD
import os, copy, operator, time
import pandas as pd
import math, itertools

class DirectoryHandlerBase:
    def __init__(self, 
                 load_data_abs_dir: str,
                 data_file_name: str,
                 vocab_file_name: Optional[str] = None,
                 load_config_abs_dir: Optional[str] = None,
                 load_ckpt_abs_path: Optional[str] = None,
                 output_abs_dir: Optional[str] = None,
                 create_run_under_abs_dir: Optional[str] = None, # will create a new output_dir if output_abs_dir is None
                 training_name: Optional[str] = None, 
                 postfix: Optional[str] = None,
                 ):
        """
        Args:
            `cwd` (str): used only for specifying the dir to store the run, this can be overwritten by `output_abs_dir`
            `data_abs_dir` (str): the absolute path to the directory containing `data_file_name` and `vocab_file_name`
            `data_file_name` (str): the name of the data file under `data_abs_dir`
            `vocab_file_name` (str): the name of the vocab file under `data_abs_dir`
            `output_abs_dir` (str): the absolute path to the output directory. If not set, it will be set by the call to the `set_output_dir` function by the TrainingManager
            `load_run_abs_dir` (str): the absolute path to the directory containing the configurations and the checkpoint to load
            `load_ckpt_name` (str): the name of the checkpoint to load under `load_run_abs_dir`
        """
        self.load_data_abs_dir = load_data_abs_dir
        self.data_file_name = data_file_name
        self.vocab_file_name = vocab_file_name
        self.load_config_abs_dir = load_config_abs_dir
        self.load_ckpt_abs_path = load_ckpt_abs_path
        self.output_abs_dir = output_abs_dir
        self.create_run_under_abs_dir = create_run_under_abs_dir
        self.training_name = training_name
        self.postfix = None
    
    @classmethod
    def load_from_file(cls, path: str):
        return cls(**clever_load(path))
    
    def save_to_file(self, path: str):
        # get a dictionary of the attributes
        clever_save(self.__dict__, path)
    
    @property
    def data_dir(self):
        return self.load_data_abs_dir
    
    @property
    def data_path(self):
        return os.path.join(self.load_data_abs_dir, self.data_file_name)
    
    @property
    def vocab_path(self):
        return os.path.join(self.load_data_abs_dir, self.vocab_file_name) if self.vocab_file_name is not None else None
    
    @property
    def load_config_dir(self):
        return self.load_config_abs_dir

    @property
    def load_ckpt_path(self):
        return self.load_ckpt_abs_path
    
    @property
    def output_dir(self):
        if self.output_abs_dir is None:
            raise ValueError("output_abs_dir is not set yet. Please call TrainingManager to set the output directory.")
        return self.output_abs_dir 

    @property
    def output_config_dir(self):
        return os.path.join(self.output_abs_dir, "configurations")
    
    @property
    def output_dirhandler_path(self):
        return os.path.join(self.output_config_dir, "dirhandler.yaml")
    
    @property
    def name_with_postfix(self):
        if self.postfix is None or self.training_name is None:
            raise ValueError("postfix or training_name is not set yet. Please call TrainingManager to set the output directory.")
        return self.training_name + '-' + self.postfix
    
    def set_output_dir(self, training_name_suggest: str, seed: int):
        if self.training_name is None:
            self.training_name = training_name_suggest
        
        # if self.postfix is None:
        current_time = time.strftime("%m%d-%H%M%S")
        self.postfix = f'seed{seed}-{current_time}'
        
        if self.output_abs_dir is None:
            # create a new output directory with the given training_name_suggest
            self.output_abs_dir = os.path.join(self.create_run_under_abs_dir, "run", self.name_with_postfix)
            
        if not os.path.exists(self.output_abs_dir):
            os.makedirs(self.output_abs_dir)
            print(f"Create output directory: {self.output_abs_dir}")
        else:
            print(f"Output directory already exists: {self.output_abs_dir}, using the existing directory.")

class Vocab:
    def __init__(self, input: Union[list, dict]):
        if isinstance(input, list):
            self.vocab = {}
            if "<bos>" not in input:
                input.append("<bos>")
            if "<eos>" not in input:
                input.append("<eos>")
            if "<pad>" not in input:
                input.append("<pad>")
            # create a vocab from the tokens
            self.vocab = {token: idx for idx, token in enumerate(input)}
        elif isinstance(input, dict):
            self.vocab = input
            # add special tokens if not already present
            if "<bos>" not in self.vocab:
                self.vocab["<bos>"] = len(self.vocab)
            if "<eos>" not in self.vocab:
                self.vocab["<eos>"] = len(self.vocab)
            if "<pad>" not in self.vocab:
                self.vocab["<pad>"] = len(self.vocab)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
    @classmethod
    def load_from_file(cls, path: str):
        return cls(clever_load(path))

    def __len__(self):
        return len(self.vocab)
    
    def __getitem__(self, key):
        return self.vocab[key]
    
    def __contains__(self, key):
        return key in self.vocab
    
    def __iter__(self):
        return iter(self.vocab)
    
    def save_to_file(self, path: str):
        clever_save(self.vocab, path)
        
    def get_word_ls(self):
        return list(self.vocab.keys())

class ConfigBase(EasyDict):
    """
    Base class for configuration. To adapt to your own setting, you can override prepare() function while super().prepare() at the beginning.
    """
    def __init__(self, config_dir: Optional[str] = None):
        """
        If config_dir is not None, the configurations in config_dir will be loaded as well as the default configurations.
        """
        super().__init__()
        self.model_config = EasyDict()
        self.train_config = EasyDict()
        self.data_config = EasyDict()
        self.probe_config = EasyDict()
        
        # get current file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # load default configurations
        self.model_config.update_from_file(os.path.join(current_dir, "configurations", "model_config_default.yaml"))
        self.train_config.update_from_file(os.path.join(current_dir, "configurations", "train_config_default.yaml"))
        self.data_config.update_from_file(os.path.join(current_dir, "configurations", "data_config_default.yaml"))
        self.probe_config.update_from_file(os.path.join(current_dir, "configurations", "train_config_default.yaml"))
        self.probe_config.update_from_file(os.path.join(current_dir, "configurations", "probe_config_default.yaml"))
        
        if config_dir is not None:
            # if config_dir + "model_config.yaml" exists, update model_config
            if os.path.exists(os.path.join(config_dir, "model_config.yaml")):
                self.model_config.update_from_file(os.path.join(config_dir, "model_config.yaml"))
            # if config_dir + "train_config.yaml" exists, update train_config
            if os.path.exists(os.path.join(config_dir, "train_config.yaml")):
                self.train_config.update_from_file(os.path.join(config_dir, "train_config.yaml"))
            # if config_dir + "data_config.yaml" exists, update data_config
            if os.path.exists(os.path.join(config_dir, "data_config.yaml")):
                self.data_config.update_from_file(os.path.join(config_dir, "data_config.yaml"))
            if os.path.exists(os.path.join(config_dir, "probe_config.yaml")):
                self.probe_config.update_from_file(os.path.join(config_dir, "probe_config.yaml"))
        # sync
        self.prepare()

    @final 
    def save_to_dir(self, save_dir: str):
        """
        Save the configuration to data_config.yaml, model_config.yaml, and train_config.yaml under configurations folder in save_dir.
        """
        self.data_config.save_to_file(os.path.join(save_dir, "data_config.yaml"))
        self.model_config.save_to_file(os.path.join(save_dir, "model_config.yaml"))
        self.train_config.save_to_file(os.path.join(save_dir, "train_config.yaml"))
        self.probe_config.save_to_file(os.path.join(save_dir, "probe_config.yaml"))
    
    @final
    def prepare(self):
        """
        Customized hook for prepare. You can override this function to add more configurations.
        """
        pass

    def override(self, kwargs: dict, verbose: bool = False):
        """
        Override the configurations with the given kwargs.
        There are two ways to override the configurations:
        
        - If the key in kwargs has the prefix of the configuration, e.g., `model_config.dim`, the corresponding key in the corresponding configuration will be updated.
        - If the key in kwargs does not have the prefix of the configuration, the key will be added to all the sub-configurations by running the `setattr_with_string` function. An example of key of this kind is `cosine_scheduler_config.warmup_steps`.
        """
        # first check if the kwargs start with self.keys(), if so update the corresponding key
        for key, var in kwargs.items():
            count = 0
            if key.startswith(tuple(self.keys())):
                self.update({key: var}) # The key already has the prefix, e.g., 'model_config.dim'
            else:
                # if not, check if the key is in the nested dict, if two nested dicts have the same key, all the keys will be updated while raise a warning message but do not stop the process
                for k, v in self.items():
                    if isinstance(v, EasyDict):
                        # if key in v.keys():
                        v.setattr_with_string(key, var)
                        count += 1
                if count > 1 and verbose:
                    print(f"Warning: {key} will be added to multiple configs. By default, all the keys will be updated.")
        self.prepare()

class PipelineBase(lightning.LightningModule):
    """
    PipelineBase is a base class for creating a training pipeline using PyTorch Lightning. 
    It provides methods for configuring optimizers, learning rate schedulers, and handling 
    training, validation, and test steps. It also includes hooks for custom behavior at 
    various stages of the training process.
    Attributes:
        train_config (EasyDict): Configuration for training.
        training_model (nn.Module): The model to be trained.
        loss_p_model (nn.Module): The primary loss model.
        loss_n_model (Optional[nn.Module]): The secondary loss model, if any.
        last_epoch (int): Tracks the last epoch to detect changes in the current epoch.
    Methods:
        from_existing_obj(cls, existing_pipeline: 'PipelineBase'):
            Initialize from an existing PipelineBase object.
        configure_optimizers():
            Configure the optimizer and learning rate scheduler.
        lr_scheduler_step(scheduler: LRSchedulerTypeUnion, metric: Any) -> None:
            Step the learning rate scheduler.
        _epoch_end_hook():
            Force the epoch end hook to be called at the end of each epoch.
        _mask_select(x, mask):
            Select the masked position of x.
        training_step(batch, batch_idx):
            Perform a training step.
        validation_step(batch, batch_idx):
            Perform a validation step.
        test_step(batch, batch_idx):
            Perform a test step.
        _Step(batch, batch_idx, step_type: str):
            Perform a forward pass and calculate losses.
        on_before_optimizer_step(optimizer: torch.optim.Optimizer):
            Called before the optimizer step.
        training_step_end(training_step_outputs):
            Handle the end of a training step.
        validation_step_end(validation_step_outputs):
            Handle the end of a validation step.
        test_step_end(test_step_outputs):
            Handle the end of a test step.
        epoch_end_hook_fn():
            Custom behavior at the end of each epoch.
    """
    def __init__(
        self, 
        train_config: EasyDict, 
        training_model: nn.Module,
        loss_p_model: nn.Module,
        loss_n_model: Optional[nn.Module] = None,
    ):
        super(PipelineBase, self).__init__()
        self.train_config = train_config
        self.training_model = training_model
        self.loss_p_model = loss_p_model
        self.loss_n_model = loss_n_model
        self.loss_n_scale = getattr(self.train_config, "loss_n_scale", 0.0)
        self.last_epoch = -1 # to track is there is a change in self.current_epoch for calling on_my_epoch_end
    
    # initialize from an existing PipelineBase object
    @classmethod
    def from_existing_obj(cls, existing_pipeline: 'PipelineBase'):
        return cls(
            config=existing_pipeline.config,
            training_model=existing_pipeline.training_model,
            loss_p_model=existing_pipeline.loss_p_model,
            loss_n_model=existing_pipeline.loss_n_model,
        )

    ## --------- default methods --------- ##
    def configure_optimizers(self):
        # Configure the optimizer.
        optimizer_dict = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'RMSprop': torch.optim.RMSprop,
            'Shampoo': Shampoo,
            'signSGD': signSGD,
            'normSGD': normSGD,
            'NormalizeSGD': NormalizeSGD,
        }
        optimizer_name = self.train_config.optimizer
        if optimizer_name not in optimizer_dict.keys():
            raise ValueError(f"Optimizer {optimizer_name} is not implemented!")
        else:
            optimizer = optimizer_dict[optimizer_name](
                self.parameters(),
                **self.train_config[f'{optimizer_name}_optimizer_config']
            )
            
        # Configure the learning rate scheduler.
        if self.train_config.lr_scheduler == "cosine":
            cosine_scheduler_config = self.train_config.cosine_scheduler_config
            scheduler = CosineAnnealingWarmup(
                optimizer=optimizer,
                warmup_steps=cosine_scheduler_config.warmup_steps,
                learning_rate=self.train_config.learning_rate,
                min_lr=cosine_scheduler_config.min_lr,
                lr_decay_steps=cosine_scheduler_config.lr_decay_steps,
            )
        elif self.train_config.lr_scheduler == "step":
            StepLR_config = self.train_config.StepLR_scheduler_config
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=StepLR_config.step_size,
                gamma=StepLR_config.gamma,
            )
        else:
            # use no scheduler
            scheduler = None
        if scheduler is not None:
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer
    
    def lr_scheduler_step(
            self,
            scheduler: LRSchedulerTypeUnion,
            metric: Any,
    ) -> None:
        scheduler.step()
    
    @final
    def _epoch_end_hook(self):
        """ 
        force the `on_my_epoch_end()` to be called at the end of each epoch, user can then override the `on_my_epoch_end()` function
        """
        if self.current_epoch != self.last_epoch: # detect change in the current_epoch
            self.last_epoch = self.current_epoch
            if self.current_epoch != 0:
                self.epoch_end_hook_fn()
    
    @final
    def _mask_select(self, x, mask):
        """
        select the masked position of `x` to compute the loss.
        """
        if mask is None:
            return x
        
        return x[mask] # select the first dimension of x where mask is True
        # if x.dim() == mask.dim() + 1:
        #     x_reshaped = x.view(-1, x.size(-1))
        #     return x_reshaped[mask.view(-1)]
        # else:
        #     assert x.dim() == mask.dim(), "The dimension of x and mask should be broadcastable"
        #     return torch.masked_select(x, mask)
        
    @final
    def training_step(self, batch, batch_idx):
        ## --------- on_my_epoch_end_hook --------- ##
        self._epoch_end_hook()
        
        loss_p, loss_n, output = self._Step(batch, batch_idx, "train")

        # if any of the three is None, return loss 0.0
        if loss_p is None or loss_n is None or output is None:
            return 0.0

        ## --------- log parameter norm --------- ##
        param_norm = 0.0
        for parameter in self.training_model.parameters():
            param_norm += torch.linalg.norm(parameter)
        self.log("param_norm", param_norm, prog_bar=True, logger=True, batch_size=self.len_batch(batch))

        loss_n_scale = self.loss_n_scale
        step_dict = {'loss': (loss_p + loss_n * loss_n_scale) / (1.0 + loss_n_scale), 'loss_p':loss_p, 'loss_n':loss_n, 'output':output, 'batch':batch}
        self.training_step_end(step_dict)
        
        return step_dict
    
    @final
    def validation_step(self, batch, batch_idx):
        """
        The validation step for the model where the model is automatically put in eval mode.

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_
        """
        # forward pass
        loss_p, loss_n, output = self._Step(batch, batch_idx, "val")
        
        # if any of the three is None, return loss 0.0
        if loss_p is None or loss_n is None or output is None:
            return 0.0
        
        loss_n_scale = self.loss_n_scale
        step_dict = {'loss': loss_p + loss_n * loss_n_scale, 'loss_p':loss_p, 'loss_n':loss_n, 'output':output, 'batch':batch}
        self.validation_step_end(step_dict)
        return step_dict
    
    @final
    def test_step(self, batch, batch_idx):
        """
        The test step for the model where the model is automatically put in eval mode.
        
        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_
        """
        # forward pass
        loss_p, loss_n, output = self._Step(batch, batch_idx, "test")

        # if any of the three is None, return loss 0.0
        if loss_p is None or loss_n is None or output is None:
            return 0.0
        
        loss_n_scale = self.train_config.loss_n_scale
        step_dict = {'loss': loss_p + loss_n * loss_n_scale, 'loss_p':loss_p, 'loss_n':loss_n, 'output':output, 'batch':batch}
        self.test_step_end(step_dict)
        return step_dict

    def _unpack_batch(self, batch):
        """
        Unpacks a batch of data into its components.

        Parameters:
            batch (tuple): A tuple containing the batch data. The batch can have:
                - 2 elements: (x, y)
                - 3 elements: (x, y, mask)
                - More than 3 elements: (x, y, mask, batch_info)

        Returns:
            tuple: A tuple containing:
                - x: The input data.
                - y: The target data.
                - mask: A mask tensor indicating valid data points.
                - batch_info: Additional batch information if present, otherwise None.
        """
        if "prompt" in batch.keys():
            x = batch["prompt"]
        else:
            raise ValueError("The batch should contain 'prompt' key.")
        if "label" in batch.keys():
            y = batch["label"]
        else:
            raise ValueError("The batch should contain 'label' key.")
        mask = batch.get("mask", torch.ones(x.shape[:2], dtype=torch.bool))
        probe_label = batch.get("probe_label", None)
        probe_mask = batch.get("probe_mask", None)
        batch_info = batch.get("batch_info", None)
        return (x, y, mask), (probe_label, probe_mask), batch_info
    
    def len_batch(self, batch):
        x = batch["prompt"]
        return len(x)
        
    ## ----------------- Cumstomized hooks ----------------- ##
    def _Step(self, batch, batch_idx, step_type: Optional[str] = None):
        ## --------- forward pass --------- ##
        
        # print("batch", batch)
        train_batch, _, _ = self._unpack_batch(batch)
        x, y, mask = train_batch
        # x (batch_size, seq_len, Optional)
        # y (batch_size, seq_len, Optional)
        # mask (batch_size, seq_len)

        output = self.training_model(x)

        # compute the loss for the masked position
        y_msk_p = self._mask_select(y, mask)
        output_msk_p = self._mask_select(output, mask)
        loss_p = self.loss_p_model(output_msk_p, y_msk_p)

        # compute the loss for the non-masked position
        y_msk_n = self._mask_select(y, ~mask)
        output_msk_n = self._mask_select(output, ~mask)

        if len(y_msk_n) > 0 and self.loss_n_model is not None:
            loss_n = self.loss_n_model(output_msk_n, y_msk_n)
        else:
            loss_n = 0.0

        ## --------- log training loss --------- ##
        if step_type is not None:
            if self.loss_n_model is not None:
                self.log(step_type + "_loss_n", loss_n, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
                self.log(step_type + "_loss_p", loss_p, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
            self.log(step_type + "_loss", loss_p + loss_n, prog_bar=True, on_epoch=True, logger=True, batch_size=self.len_batch(batch)) # should always log the total loss as 'val_loss' is used for ckpt saving
            # don't change the log name step_type + '_loss' as it is used for ckpt saving

        return loss_p, loss_n, output
    
    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer):
        """ 
        This function is called before the optimizer step.
        You can override this function to do something before the optimizer step.

        Args:
            optimizer (torch.optim.Optimizer): the optimizer
        """
        norms = lightning.pytorch.utilities.grad_norm(self.training_model, norm_type=2)
        self.log_dict(norms)

    def training_step_end(self, training_step_outputs):
        """
        Designed for 1. logging, 2. visualization, 3. saving, 4. etc.
 
        You can override this function yourself but remember to include the following lines.
        ```
        loss = training_step_outputs['loss']
        loss_p = training_step_outputs['loss_p']
        loss_n = training_step_outputs['loss_n']
        output = training_step_outputs['output']
        batch = training_step_outputs['batch']
        x, y, mask,  = batch
        ```
        Make sure that the return is the loss you want to optimize
        """
        # return training_step_outputs['loss_p'] + training_step_outputs['loss_n']
    
    def validation_step_end(self, validation_step_outputs):
        return self.training_step_end(validation_step_outputs)
    
    def test_step_end(self, test_step_outputs):
        return self.training_step_end(test_step_outputs)

    def epoch_end_hook_fn(self):
        """
        Do something at the end of each epoch. Feel free to override this function.
        """
        pass

    def create_hook_fn(self, 
                       model_to_hook_str: str,
                       tensor_to_hook_str: str, 
                       storage_dict: EasyDict):
        """return a hook function that can be used to hook a tensor from a model and store it in a storage_dict.

        Args:
            model_to_hook_str (str): The name of the model to hook. This should be the name of the model variable in pipeline.train_model.
            tensor_to_hook_str (str): The name of the tensor to hook. This should be the name of the tensor in the model output/input. Check the intermediate results' keys in the forward method of the model.
            storage_dict (EasyDict): _description_
        """
        def hook_fn(module, input, output):
            ## --------- change the probe model input here --------- ##
            if isinstance(output, tuple):
                direct_output, intermediate_dict = output
            else:
                direct_output = output
                intermediate_dict = None
            # combine model_to_hook_str and tensor_to_hook_str with a dot
            keyword = f"{model_to_hook_str}.{tensor_to_hook_str}"
            if tensor_to_hook_str == "output":
                storage_dict.setattr_with_string(keyword, direct_output)
            elif tensor_to_hook_str == "input":
                if isinstance(input, tuple):
                    storage_dict.setattr_with_string(keyword, input[0])
                else: 
                    storage_dict.setattr_with_string(keyword, input)
            else:
                storage_dict.setattr_with_string(keyword, intermediate_dict[tensor_to_hook_str])
        return hook_fn
    
    def add_forward_hooks(self, hook_target_key, model): 
        # ---- initialize the dictionary for hooking positions ---- #
        hook_dict = EasyDict({})
        for added_key in hook_target_key:
            hook_dict.setattr_with_string(added_key, None)
        storage_dict = EasyDict(copy.deepcopy(hook_dict))
        
        ## --------- autohook for sae model --------- ##
        num_hook = 0
        for key, value in hook_dict.flatten().items():
            # split the key into the model to hook and the specific tensor to hook by the last dot. Example: model.blocks.layer_0.attn.output -> model.blocks.layer_0.attn, output
            model_to_hook_str, tensor_to_hook_str = key.rsplit(".", 1)
            model_to_hook = operator.attrgetter(model_to_hook_str)(model)
            # print('key:', key)
            hook_dict.setattr_with_string(
                key, model_to_hook.register_forward_hook(self.create_hook_fn(model_to_hook_str, tensor_to_hook_str, storage_dict))
            )
            num_hook += 1
            
        return hook_dict, storage_dict, num_hook
    
    
class DataModuleBase(lightning.LightningDataModule):
    """
    DataModuleBase is a base class for creating a data module using PyTorch Lightning.
    It provides methods for setting up the data, creating data loaders, and transforming batches.
    Attributes:
        data_config (EasyDict): Data configuration.
        dir_handler (DirectoryHandlerBase): Directory handler for loading and saving data.
        vocab (Optional[Vocab]): Vocabulary for the data.
        data_train (Any): Training data.
        data_val (Any): Validation data.
        data_test (Any): Test data.
        data_predict (Any): Prediction data.
    Methods:
        setup(stage: str):
            Setup the data module for the given stage.
        train_dataloader():
            Create the training data loader.
        val_dataloader():
            Create the validation data loader.
        test_dataloader():
            Create the test data loader.
        predict_dataloader():
            Create the prediction data loader.
        on_before_batch_transfer(batch, dataloader_idx: int):
            Called before transferring the batch to the device.
        state_dict():
            Return the state dictionary.
        load_state_dict(state_dict):
            Load the state dictionary.
        prepare_data():
            Download and prepare the data.
        train_val_test_split(data):
            Split the data into training, validation, and test sets.
        transform_batch(batch, dataloader_idx):
            Transform the batch before sending it to the model.
        transfer_batch_to_device(batch, device, dataloader_idx):
            Transfer the batch to the device.
        """
    def __init__(self, data_config: EasyDict, dir_handler: DirectoryHandlerBase):
        """
        Args:
            data_config (EasyDict): data configuration
            dir_handler (DirectoryHandlerBase): directory handler
        """
        super().__init__()
        self.data_config = data_config
        self.dir_handler = dir_handler
        
        if not os.path.exists(self.dir_handler.vocab_path):
            self.vocab = None
        else:
            self.vocab = Vocab(clever_load(self.dir_handler.vocab_path))

    @final
    def setup(self, stage: str):
        if stage == "fit":
            data_full = clever_load(self.dir_handler.data_path)
            self.data_train, self.data_val, self.data_test = self.train_val_test_split(data_full)
        
        if stage == "test":
            # if self does not have data_test, then use train_val_test_split to split data
            if not hasattr(self, "data_test"):
                data_full = clever_load(self.dir_handler.data_path)
                self.data_train, self.data_val, self.data_test = self.train_val_test_split(data_full)

        if stage == "predict":
            self.data_predict = clever_load(self.dir_handler.data_path)

    def train_dataloader(self):
        num_workers = self.data_config.num_workers * torch.cuda.device_count()
        if os.name == 'nt':
            num_workers = 0
        return DataLoader(self.data_train, 
                          batch_size=self.data_config.batch_size, 
                          collate_fn=lambda x: x,
                          shuffle=True,
                          num_workers=num_workers)
    
    def val_dataloader(self):
        num_workers = max(4, self.data_config.num_workers) * torch.cuda.device_count()
        if os.name == 'nt':
            num_workers = 0
        return DataLoader(self.data_val, 
                          batch_size=self.data_config.batch_size, 
                          collate_fn=lambda x: x,
                          shuffle=False,
                          num_workers=num_workers)
    
    def test_dataloader(self):
        num_workers = max(2, self.data_config.num_workers) * torch.cuda.device_count()
        if os.name == 'nt':
            num_workers = 0
        return DataLoader(self.data_test, 
                          batch_size=self.data_config.batch_size, 
                          collate_fn=lambda x: x,
                          shuffle=False,
                          num_workers=num_workers)
    
    def predict_dataloader(self):
        num_workers = self.data_config.num_workers * torch.cuda.device_count()
        if os.name == 'nt':
            num_workers = 0
        return DataLoader(
            self.data_predict, 
            batch_size=self.data_config.batch_size, 
            collate_fn=lambda x: x,
            shuffle=False, 
            num_workers=num_workers)
    
    @final
    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        """
        The functionality of this function is deferred to the method transfer_batch_to_device. Don't override this function.
        """
        return self.transform_batch(batch, dataloader_idx)

    ## ----------------- Cumstomized functions ----------------- ##
    def state_dict(self):
        """
        Customized state_dict function. You can override this function to save more states.
        Usually, you don't need to do so unless you define more states in the __init__ function.
        """
        state = {}
        return state

    def load_state_dict(self, state_dict):
        """
        Customized load_state_dict function. You can override this function corresponding to the state_dict function.
        """
        pass

    def prepare_data(self):
        """ 
        download, do some global things here, but you can't return anything
        """
        pass 

    def train_val_test_split(self, data):
        """
        Split data into train, validation, and test sets. Feel free to override this function. You should return data_train, data_val, data_test. Here is an example:
        
        ```python
        data_train, data_test = random_split(data, [int(0.9*len(data)), len(data)-int(0.9*len(data))])
        data_train, data_val = random_split(data_train, [int(0.9*len(data_train)), len(data_train)-int(0.9*len(data_train))])
        ```
        """
        raise NotImplementedError("You should implement the train_val_test_split function.")
        return data_train, data_val, data_test
    
    def transform_batch(self, batch, dataloader_idx):
        """
        Transform batch before sending to the model
        Here, users are expected to transform the batch into three tensors: 
            x (batch_size, seq_len, Optional),
            y (batch_size, seq_len, Optional), 
            (Optional) mask (batch_size, seq_len) indicating where the loss should be computed
        You should return a tuple (x, y, mask) or (x, y) if mask not needed.
        """
        # raise a warning to remind the user to implement the function
        raise NotImplementedError("You should implement the transform_batch function.")
        return batch
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Usually you do not need to adjust this function as long as you have the batch in the right format in on_before_batch_transfer
        Supported batch types:
            torch.Tensor or anything that implements .to(…)
            list
            dict
            tuple
        If you have a very special data format, you can override this function and return a tuple for batch.

        Here is an example:
        ```python
        x, y, msk, batch_info = batch
        x = x.to(device)
        y = y.to(device)
        msk = msk.to(device)
        return x, y, msk, batch_info
        ```
        """
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(device)
            elif isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, torch.Tensor):
                        val[k] = v.to(device)
        return batch


    

class ProbePipelineBase(PipelineBase):
    def __init__(self, 
                 probe_config: EasyDict, 
                #  training_model: nn.Module,
                 probe_layer: nn.Module,
                 probe_loss_model: nn.Module,
                 pipeline: PipelineBase, # suppose you have a pipeline that inherits from PipelineBase, then pass it here
                 added_probe_target_key: Optional[str] = None, 
                 added_vis_target_key: Optional[str] = None,
                 pos_label: list = None,
                 ) -> None:
        """Note that pipeline here is not attached to any trainer. So all the hooks are not activated, including the logger.

        Args:
            probe_config (EasyDict): _description_
            probe_layer (nn.Module): _description_
            probe_loss_model (nn.Module): _description_
            pipeline (PipelineBase): _description_
            thenpassithereadded_probe_target_key (Optional[str], optional): _description_. Defaults to None.
            added_vis_target_key (Optional[str], optional): _description_. Defaults to None.
        """
        super(ProbePipelineBase, self).__init__(
            train_config=probe_config, 
            training_model=probe_layer, 
            loss_p_model=probe_loss_model)
        # initialize the CustomPipeline
        self.added_probe_target_key = added_probe_target_key
        self.added_vis_target_key = added_vis_target_key
        self.probe_loss_model = probe_loss_model
        self.pipeline = pipeline
        
        # ---- initialize the dictionary for hooking positions ---- #
        
        self.probe_dict, self.probe_storage_dict, self.num_probe_hook = self.add_forward_hooks(self.added_probe_target_key, self.pipeline.training_model)
        
        self.vis_dict, self.vis_storage_dict, self.num_vis_hook = self.add_forward_hooks(self.added_vis_target_key, self.pipeline.training_model)
        # probe_dict = EasyDict({})
        # for added_key in self.added_probe_target_key:
        #     probe_dict.setattr_with_string(added_key, None)
        # self.probe_storage_dict = EasyDict(copy.deepcopy(probe_dict))
        
        # ## ---- initialize the dictionary for hooking attention ---- ##
        # vis_dict = EasyDict({})
        # for added_key in self.added_vis_target_key:
        #     vis_dict.setattr_with_string(added_key, None)
        # self.vis_storage_dict = EasyDict(copy.deepcopy(vis_dict))

        # ## --------- autohook for probe model --------- ##
        # self.num_probe_hook = 0
        # for key, value in probe_dict.flatten().items():
        #     # split the key into the model to hook and the specific tensor to hook by the last dot. Example: model.blocks.layer_0.attn.output -> model.blocks.layer_0.attn, output
        #     model_to_hook_str, tensor_to_hook_str = key.rsplit(".", 1)
        #     model_to_hook = operator.attrgetter(model_to_hook_str)(self.pipeline.training_model)
        #     # print('key:', key)
        #     probe_dict.setattr_with_string(
        #         key, model_to_hook.register_forward_hook(self.create_hook_fn(model_to_hook_str, tensor_to_hook_str, self.probe_storage_dict))
        #     )
        #     self.num_probe_hook += 1
            
        # ## --------- autohook for vis model --------- ##
        # for key, value in vis_dict.flatten().items():
        #     model_to_hook_str, tensor_to_hook_str = key.rsplit(".", 1)
        #     model_to_hook = operator.attrgetter(model_to_hook_str)(self.pipeline.training_model)
        #     # print('key:', key)
        #     vis_dict.setattr_with_string(
        #         key, model_to_hook.register_forward_hook(self.create_hook_fn(model_to_hook_str, tensor_to_hook_str, self.vis_storage_dict))
        #     )
        
        # self.probe_dict = probe_dict
        # self.vis_dict = vis_dict # for later releasing the memory
        
        self.channel_loss_logger = []
        self.pos_label = pos_label
        print("Number of probe hooks added:", self.num_probe_hook)

    @property
    def probe_layer(self):
        """Give a reference to the training model. The name probe_layer is easier to understand than training_model.
        """
        return self.training_model
        
    
    
    def supply_hidden_state_tensor(self, pos: torch.Tensor):
        """
        """
        # check if pos is a two-dimensional tensor
        if pos.dim() != 2:
            raise ValueError("pos should be a two-dimensional tensor")
        row_sums = torch.sum(pos, dim = 1)
        if not torch.all(row_sums == row_sums[0]):
            raise ValueError("pos should be a two-dimensional tensor with the same sum of each row")
        
        # flatten the forward_return_dict
        probe_storage_dict = self.probe_storage_dict.flatten()
        
        # get the values of the forward_return_dict_flattened
        probe_storage_ls = list(probe_storage_dict.values()) # a list of tensors, each tensor is of shape (batch_size, seq_len, hidden_size)
        
        probe_storage_tensor = torch.stack(probe_storage_ls, dim = -1) # shape: (batch_size, seq_len, hidden_size, num_probe_hook)
        
        probe_storage_msk_tensor = self._mask_select(probe_storage_tensor, pos) # shape: (batch_size * msk_seq_len, hidden_size, num_probe_hook)

        batch_size = probe_storage_tensor.shape[0]
        probe_storage_msk_tensor = probe_storage_msk_tensor.reshape(batch_size, -1, *probe_storage_msk_tensor.shape[1:])
        # shape: (batch_size, msk_seq_len, hidden_size, num_probe_hook)

        # swap the dimensions
        probe_storage_msk_tensor = probe_storage_msk_tensor.permute(0, 3, 1, 2) # shape: (batch_size, num_probe_hook, msk_seq_len, hidden_size)
        
        # hidden_state_tensor = probe_storage_msk_tensor.reshape(probe_storage_msk_tensor.shape[0], -1, probe_storage_msk_tensor.shape[-1]) # shape: (batch_size, num_probe_hook * msk_seq_len, hidden_size)

        return probe_storage_msk_tensor # shape: (batch_size, num_probe_hook,  msk_seq_len, hidden_size)

    def _reshape_probe_state_and_label(self, 
                                       probing_output, 
                                       probe_label, 
                                       in_channel_size_ls,
                                        out_channel_size_ls,
                                        total_channel_size_ls):
        """
        Reshape the probing_output and probe_label to the same shape for calculating the loss.
        
        probing_output: (batch_size, [num_probe_hook, msk_seq_len], *out_channel_size_ls, probe_output_size) -> (batch_size, probe_output_size, num_probe_hook, msk_seq_len, *out_channel_size_ls)
        
        probe_label: (batch_size, *out_channel_size_ls) -> (batch_size, [num_probe_hook, msk_seq_len], *in_channel_size_ls, *out_channel_size_ls)
        
        [num_probe_hook, msk_seq_len] = *in_channel_size_ls
        """
        
        # move the last dimension to the dimension 1
        probing_output = probing_output.permute(0, -1, *range(1, len(total_channel_size_ls) + 1)) # shape: (batch_size, probe_output_size, *in_channel_size_ls, *out_channel_size_ls)
        
        assert probe_label.shape[-len(out_channel_size_ls):] == out_channel_size_ls, f"The last dimensions of probe_label should be the same as out_channel_size_ls{out_channel_size_ls}, but we find {probe_label.shape[-len(out_channel_size_ls):]}"
        
        label_intrinsic_size_ls = probe_label.shape[:-len(out_channel_size_ls)]
        
        # add len(in_channel_size_ls) new dimensions previous to the last len(out_channel_size_ls) dimensions
        for i in range(len(in_channel_size_ls)):
            probe_label = probe_label.unsqueeze(-len(out_channel_size_ls)-1)
            # expand the added dimension to the size of in_channel_size_ls[i]
        # shape: (batch_size, ..., 1, ..., 1, *out_channel_size_ls)
        probe_label = probe_label.expand(*label_intrinsic_size_ls, *in_channel_size_ls, *out_channel_size_ls) # shape: (label_intrinsic_size_ls, *in_channel_size_ls, *out_channel_size_ls)
        return probing_output, probe_label
    
    def _Step(self, batch, batch_idx, step_type: str):
        ## --------------- forward pass --------------- ##
        self.pipeline.training_model.eval()
        with torch.no_grad():
            loss_p, loss_n, output = self.pipeline._Step(
                batch, 
                batch_idx, 
                step_type='predict', # do not log the loss because the pipeline is not attached to a trainer
                )
            
        _, probe_batch, _ = self._unpack_batch(batch)
        probe_loc_mask = probe_batch[1]
        probe_label = probe_batch[0]
        
        hidden_state_tensor = self.supply_hidden_state_tensor(probe_loc_mask) # shape: (batch_size, num_probe_hook, msk_seq_len, hidden_size)
        
        ## --------------- probing model --------------- ##
        probing_output = self.probe_layer(hidden_state_tensor) # shape: (batch_size, num_probe_hook, msk_seq_len, *out_channel_size_ls, probe_output_size)
        
        in_channel_size_ls = hidden_state_tensor.shape[1:-1]
        total_channel_size_ls = probing_output.shape[1:-1]
        out_channel_size_ls = total_channel_size_ls[len(in_channel_size_ls):]
        
        probing_output_ext, probe_label_ext = self._reshape_probe_state_and_label(probing_output, 
                                            probe_label, 
                                            in_channel_size_ls, out_channel_size_ls, total_channel_size_ls)
        
        probe_loss = self.probe_loss_model(probing_output_ext, probe_label_ext)

        ## -------------- log the loss -------------- ##

        return probe_loss, probing_output_ext, probe_label_ext
    
    
    def training_step(self, batch, batch_idx):
        ## ---------- on_my_epoch_end_hook ------------ ##
        self._epoch_end_hook()    
        
        probe_loss, _, _ = self._Step(batch, batch_idx, "train")
        
        total_channel_num = 1
        # check if probe_loss is a scalar
        if probe_loss.dim() != 0:
            # average over the first dimension
            probe_loss = probe_loss.mean(dim=0)
            # take a sum over the rest of the dimensions
            total_channel_num = probe_loss.numel()
            probe_loss = probe_loss.sum()
        
        self.log("probe_train_loss", probe_loss / total_channel_num, prog_bar=True, logger=True, batch_size=self.len_batch(batch))

        return probe_loss
    
    def validation_step(self, batch, batch_idx):
        probe_loss, _, _ = self._Step(batch, batch_idx, "val")

        # check if probe_loss is a scalar
        total_channel_num = 1
        if probe_loss.dim() != 0:
            # average over the first dimension
            probe_loss = probe_loss.mean(dim=0)
            # take a sum over the rest of the dimensions
            total_channel_num = probe_loss.numel()
            probe_loss = probe_loss.sum()
            probe_loss = probe_loss.sum()
            
        self.log("probe_val_loss", probe_loss/total_channel_num, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        return probe_loss
    
    def test_step(self, batch, batch_idx):
        probe_loss, _, _ = self._Step(batch, batch_idx, "test")

        if probe_loss.dim() != 0:
            probe_loss = probe_loss.mean(dim=0)
            
        self.channel_loss_logger.append((probe_loss, self.len_batch(batch)))
        return probe_loss.sum()
    
    def process_and_reset_channel_loss(self, pos_label: Optional[list] = None):
        """
        Process the channel loss for visualization, by default, use self.pos_label
        """
        if pos_label is None:
            pos_label = self.pos_label
        if len(self.channel_loss_logger) == 0:
            return None
        else:
            cum_channel_loss = torch.zeros(self.channel_loss_logger[0][0].shape, device=self.channel_loss_logger[0][0].device)
            cum_sample_size = 0
            for channel_loss, batch_size in self.channel_loss_logger:
                cum_channel_loss += channel_loss * batch_size
                cum_sample_size += batch_size
            channel_loss = cum_channel_loss / cum_sample_size

            # make a pd table with rows as the keys of the probe_storage_dict
            
            in_channel_size_ls = list(channel_loss.shape[0:2])
            out_channel_size_ls = list(channel_loss.shape[2:])
            
            probe_storage_dict_keys = list(self.probe_storage_dict.flatten().keys())
            
            channel_loss_df_ls = []
            ranges = [range(x) for x in out_channel_size_ls]
            for index, combination in enumerate(itertools.product(*ranges)):
                print(index, combination)
                
                index_tuple = (slice(None), slice(None), *combination)
                channel_loss_df = pd.DataFrame(channel_loss[index_tuple].detach().cpu().numpy(), index=probe_storage_dict_keys, columns=pos_label)
                
                channel_loss_df_ls.append(channel_loss_df)
            
            self.channel_loss_logger = []
            
            return channel_loss_df_ls


class SAEPipelineBase(PipelineBase):
    def __init__(self, 
                 sae_config: EasyDict, 
                 sae_layer: nn.Module,
                 pipeline: PipelineBase,
                 added_sae_target_key: Optional[str] = None,
    ):
        super(SAEPipelineBase, self).__init__(
            train_config=sae_config, 
            training_model=sae_layer, 
            loss_p_model=None)
        self.added_sae_target_key = added_sae_target_key
        self.pipeline = pipeline
        
        self.hook_dict, self.storage_dict, self.num_hook = self.add_forward_hooks(self.added_sae_target_key, self.pipeline.training_model)
        
        print("Number of sae hooks added:", self.num_sae_hook)
    
    @property
    def sae_layer(self):
        return self.training_model
    
    def supply_hidden_state_tensor(self):
        hidden_state = list(self.storage_dict.flatten().values())
        hidden_state_tensor = torch.stack(hidden_state, dim = -1) # shape: (batch_size, seq_len, hidden_size, num_sae_hook)
        hidden_state_tensor = hidden_state_tensor.reshape(-1, *hidden_state_tensor.shape[-2:]) # shape: (batch_size * seq_len, hidden_size, num_sae_hook)
        return hidden_state_tensor.permute(0, 2, 1) # shape: (batch_size * seq_len, num_sae_hook, hidden_size)
    
    def _Step(self, batch, batch_idx, step_type: str):
        ## --------------- forward pass --------------- ##
        self.pipeline.training_model.eval()
        with torch.no_grad():
            _ = self.pipeline._Step(
                batch, 
                batch_idx, 
                step_type='predict', # do not log the loss because the pipeline is not attached to a trainer
                )
        hidden_state_tensor = self.supply_hidden_state_tensor()
        
        reconstructed_loss, l1_loss, intermediate_state = self.sae_layer(hidden_state_tensor, **self.train_config)
        
        pre_act = intermediate_state["pre_act"]
        sparsity = 1 - (pre_act > 1e-3).sum() / pre_act.numel()
        avg_act_pos = pre_act[pre_act > 1e-3].mean()
        
        self.log(step_type + "_reconstructed_loss", reconstructed_loss, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        self.log(step_type + "_l1_loss", l1_loss, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        
        self.log(step_type + "_sparsity", sparsity, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        self.log(step_type + "_avg_act_pos", avg_act_pos, prog_bar=True, logger=True, batch_size=self.len_batch(batch))
        
        loss = reconstructed_loss + l1_loss
        
        return loss, 0.0, None
