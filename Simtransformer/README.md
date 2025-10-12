# Basic Format Setting

Here, we introduce some basic format in Simtransformer.

## Configuration Setting

## Data Setting

batch (tuple): A tuple containing the batch data. The batch can have:

* 2 elements: `(x, y)`
* 3 elements: `(x, y, msk)`
* More than 3 elements: `(x, y, msk, batch_info)`

Here, the shapes are given by

* `x.shape`= (batch_size, seq_len, Optional), where `Optional` can either be `None` for token index or `embedding_dim` if `x` is supplied as the embedding.
* `y.shape` =  (batch_size, seq_len, Optional), where `Optional` is usually the `logits`, i.e., log probability, or the output dimension.
* `msk.shape` = (batch_size, seq_len), a bool tensor indicating where the losses are computed.

In fact, for the position `msk` == `True`, `loss_p_model` in method **config_pipeline()** is applied and for the position `msk` == `False`, `loss_n_model` is applied. The final loss will be the the summation `loss` = `loss_p` + `loss_n`.

### Customizable Methods for Supplying a Batch in a Data

The methods related to batch are majorly in class `DataModuleBase`:

* **train_val_test_split(self, data)**: return (`data_train`, `data_val`, `data_test`)

* **transform_batch(self, batch, dataloader_idx)**: to pad the sequences in the batch and make it ready to be fed into the forward process. The return is a dictionary that should contains the keys "prompt", "label", "mask", "probe_label", "probe_mask", "batch_info". 
For only doing the training, "probe_label", "probe_mask", "batch_info" can be 

* **prepare_data()**: used for downloading dataset, or you can use it for regenerating the dataset. 

Here is an example: 

```python
class DataModule(DataModuleBase):
    def train_val_test_split(self, data):
        # split data into train, validation, and test sets by ratio 90:5:5
        data_train, data_test = random_split(data, [int(0.9*len(data)), len(data)-int(0.9*len(data))])
        data_train, data_val = random_split(data_train, [int(0.9*len(data_train)), len(data_train)-int(0.9*len(data_train))])
        return data_train, data_val, data_test
    
    def transform_batch(self, batch, dataloader_idx):
        """
        Here, each sample consists of a dictionary with "pos", "sentence" and "reasoning_path" keys.
        """
        max_seq_len = max([len(sample['sentence']) for sample in batch])
        x_tensor = torch.zeros(len(batch), max_seq_len - 1, dtype=torch.long).fill_(self.vocab["<pad>"])
        y_tensor = torch.zeros(len(batch), max_seq_len - 1, dtype=torch.long)
        batch_info = []
        msk_tensor = torch.zeros(len(batch), max_seq_len - 1, dtype=torch.bool)
        probe_msk_tensor = torch.zeros(len(batch), max_seq_len - 1, dtype=torch.bool)
        probe_label = torch.zeros(len(batch), 1, dtype=torch.long).fill_(self.vocab["<pad>"])
        
        for i, sample in enumerate(batch):
            sentence, reasoning_path, pos = sample['sentence'], sample['reasoning_path'], sample['pos']
            sentence_idx = [self.vocab[word] for word in sentence]
            sentence_idx_padded = sentence_idx + [self.vocab["<pad>"]] * (max_seq_len - len(sentence_idx))
            
            x_tensor[i, :] = copy.deepcopy(torch.tensor(sentence_idx_padded[:-1]))
            y_tensor[i, :] = copy.deepcopy(torch.tensor(sentence_idx_padded[1:]))
            
            # find the position of '<pre_rel>' and set the token next to it to be '<Q>'
            pre_rel_pos = sentence.index('<pre_rel>')
            rel_pos = pre_rel_pos + 1
            pre_dst_pos = sentence.index('<pre_dst>')
            dst_pos = pre_dst_pos + 1
            
            msk_tensor[i, torch.where(x_tensor[i, :] == self.vocab['<pre_dst>'])] = True
            
            # the following two is for probing
            probe_msk_tensor[i, [pre_rel_pos, rel_pos, pre_dst_pos]] = True

            probe_label[i, 0] = sentence_idx_padded[dst_pos]
            
            batch_info.append(sample)
            
        return EasyDict({
            "prompt": x_tensor,
            "label": y_tensor,
            "mask": msk_tensor,
            "probe_label": probe_label,
            "probe_mask": probe_msk_tensor,
            "batch_info": batch_info
        })
```

### DataModuleBase Configuration

The DataModuleBase is initialized by data_config


## Probing Model

### Channel

We treat each hidden state at each position as an `in_channel`, while each probing task as an `out_channel`. The total number of linear probes we are training is thus the product of the total number of `in_channel` and `out_channel`.

The input channel always takes the shape of (probe_hook_num, probe_pos_len), where `probe_hook_num` is the total number of probing hook added to the model, and `probe_pos_len` is the total number of locations in the prompt sequence we want to probe. The actual locations to probe is supplied by the 'probe_mask' items in the `transform_batch()` method

```
Notice: probe_label must be of shape (batch_size, *out_channel_size_ls), even if there is only one out_channel.
```

### Probing loss

If the loss model is `nn.Crossentropy`, the reduction should be set to `'none'` because we are interested in obtaining individual loss for each input/output channel