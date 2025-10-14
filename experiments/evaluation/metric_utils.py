import torch
import copy
from tqdm import tqdm
import numpy as np
from Simtransformer.simtransformer.utils import attach_hooks, EasyDict


def attach_hooks_to_model(litmodel):
    HookDict = EasyDict()

    for i in range(litmodel.model_config.num_layers):
        HookDict[f'L{i}_R0'] = f'blocks.layer_{i}.input'
        HookDict[f'L{i}_R1'] = f'blocks.layer_{i}.attn_res_output'
        HookDict[f'L{i}_R2'] = f'blocks.layer_{i}.output'
        HookDict[f'L{i}_Ain'] = f'blocks.layer_{i}.attn.input'
        HookDict[f'L{i}_Aout'] = f'blocks.layer_{i}.attn.output'
        HookDict[f'L{i}_Min'] = f'blocks.layer_{i}.mlp.input'
        HookDict[f'L{i}_Mout'] = f'blocks.layer_{i}.mlp.output'
    for i in range(litmodel.model_config.num_layers):
        HookDict[f'L{i}_attn_prob'] = f'blocks.layer_{i}.attn.attn_prob'

    # attach hooks to the TF model
    buffer, hook_handles = attach_hooks(litmodel.model.transformer, HookDict)
    return buffer, hook_handles

def compute_attention_metrics(buffer):
    # attn_score keys
    attn_score_keys = [k for k in buffer.keys() if 'attn_prob' in k]

    layerwise_max_attn_score = {k: buffer[k].max(dim=-1).values.mean().item() for k in attn_score_keys}
    avg_max_attn_score = torch.tensor(list(layerwise_max_attn_score.values())).mean()

    layerwise_attn_score_entropy = {k: -(buffer[k] * torch.log(buffer[k] + 1e-6)).sum(dim=-1).mean().item() for k in attn_score_keys}
    avg_attn_score_entropy = torch.tensor(list(layerwise_attn_score_entropy.values())).mean()

    # Entropy / log(n), where n = number of tokens
    layerwise_attn_score_normalized_netropy = {k: -(buffer[k] * torch.log(buffer[k] + 1e-6)).sum(dim=-1).mean().item() / np.log(buffer[k].shape[-1]) for k in attn_score_keys}
    avg_attn_score_normalized_entropy = torch.tensor(list(layerwise_attn_score_normalized_netropy.values())).mean()

    attention_metrics = dict(
        attn_score_entropy=avg_attn_score_entropy,
        attn_score_normalized_entropy=avg_attn_score_normalized_entropy,
        max_attn_score=avg_max_attn_score)
    return attention_metrics

def one_step_prediction(litmodel, batch, step, outputs=None, step_stack=None):

    batch = batch.to(litmodel.device)
    # print('inputs', inputs.shape)
    if hasattr(litmodel, 'nointerm') and litmodel.nointerm:
        if step == 0:
            with torch.no_grad():
                input_emb = litmodel.model.token_embedder(batch)
        else:
            input_emb = outputs
        # if input only has two dimensions, add a dimension
        if input_emb.dim() == 2:
            input_emb = input_emb.unsqueeze(0)
        outputs, logitss = litmodel.predict_step(input_emb)
        return outputs, logitss
    else:
        inputs = step_stack[-1]
        inputs.to(litmodel.device)
        # if input only has two dimensions, add a dimension
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        logitss = litmodel.predict_step(inputs)
        return None, logitss



def calc_batch_metrics(litmodel, batch, factored_tokenizer, n_steps=None, factors_slice=None, verbose=False):

    litmodel.eval()
    buffer, hook_handles = attach_hooks_to_model(litmodel)

    device = litmodel.device

    batch = batch.to(device)

    model_config = litmodel.model_config

    if factors_slice is None:
        factors_slice = slice(0, 6) # consider all factors

    with torch.no_grad():
        # process batch
        # model_config, data_config = litmodel.model_config, litmodel.data_config
        batch = litmodel.reduce_batch(copy.deepcopy(batch))
        batch, batch_label, batch_info = batch[..., :model_config.data_dim], batch[..., model_config.data_dim:model_config.data_dim + model_config.label_dim], batch[..., model_config.data_dim + model_config.label_dim:]
        # note: batch_label is final label

        # get teacher batch
        depths = batch_info[..., 0]
        if n_steps is None:
            n_steps = depths.max().item() + 1
            print('inferred max steps:', n_steps)

        batch_ext, value_avail_pos, value_to_compute_pos = litmodel.get_teacher_batch_with_depth(batch, batch_label, depths)
        # batch_ext.shape: [n_steps, batch_size, seq_len, n_factors]

        # infer which tokens are padding, and ignore them in computing accuracy
        pad_encoding = torch.tensor(factored_tokenizer.encode_string(factored_tokenizer.PAD)[0]).to(device)
        is_pad = torch.eq(batch, pad_encoding).all(dim=-1)

        # get first pad for each sample
        orig_seq_len = batch.size(1)
        seq_lens = (~is_pad).sum(dim=1)
        max_seq_len = seq_lens.max()
        if verbose:
            print('max non-pad length:', max_seq_len)
            print('orig seq len:', orig_seq_len)

        is_pad = is_pad.unsqueeze(-1).repeat(1, 1, 6)
        is_pad = is_pad[..., factors_slice]

        # calc metrics up to max_seq_len in batch
        batch = batch[:, :max_seq_len]
        batch_ext = batch_ext[:, :, :max_seq_len]
        batch_label = batch_label[:, :max_seq_len]
        is_pad = is_pad[:, :max_seq_len]

        # inputs = batch[0]
        step_stack = [batch]


        metrics =  dict(factor_acc=[], position_acc=[], fully_correct_acc=[], recorrection=[], step_final_label_full_acc=[], step_final_label_factor_acc=[], step_final_label_persymbol_avg_acc=[], step_last_var_factor_acc=[], step_last_var_fully_correct=[])

        last_mistakes = None
        outputs = None
        for step in range(0, n_steps):
            if verbose:
                print('step', step)

            # one_step_prediction
            outputs, logitss = one_step_prediction(litmodel, batch, step, outputs=outputs, step_stack=step_stack)

            preds = torch.stack([logits.argmax(-1) for logits in logitss], dim=-1) # argmax prediction for each factoro
            step_stack.append(preds)
            # print('preds', preds.shape)

            if step + 1 <= len(batch_ext) - 1:
                step_label = batch_ext[step + 1]
            else:
                step_label = batch_ext[-1]

            is_correct = (preds == step_label)[..., factors_slice]
            is_correct = torch.logical_or(is_correct, is_pad)

            # percent recorrection
            if last_mistakes is None or last_mistakes.sum() == 0:
                recorrection = torch.nan
            else:
                recorrection = torch.masked_select(is_correct, last_mistakes).float().mean().item()
            metrics['recorrection'].append(recorrection)

            factor_acc = is_correct.float().mean(axis=(0,1))
            metrics['factor_acc'].append(factor_acc)

            # accuracy by position (accounting for all factors)
            position_acc = is_correct.all(axis=-1).float().mean(axis=(0))
            # pad postion_acc with nan until orig_seq_len
            position_acc = torch.cat([position_acc, torch.full((orig_seq_len - position_acc.size(0),), fill_value=torch.nan).to(device=batch.device)])
            metrics['position_acc'].append(position_acc)

            # check whether step label is correct
            fully_correct = is_correct.all(axis=(1,2)) # sequence accuracy
            fully_correct_acc = fully_correct.float().mean()
            metrics['fully_correct_acc'].append(fully_correct_acc)

            persymbol_avg_acc = is_correct.all(axis=2).float().mean(axis=(0,1)).mean() # average accuracy per symbol
            metrics['persymbol_avg_acc'] = persymbol_avg_acc

            # check whether final label is correct (i.e., full solved)
            final_is_correct = (preds == batch_label)[..., factors_slice]
            final_is_correct = torch.logical_or(final_is_correct, is_pad)
            final_label_full_acc = final_is_correct.all(axis=(1,2)).float().mean() # sequence accuracy
            final_label_factor_acc = final_is_correct.float().mean(axis=(0,1)) # accuracy by factor, averaged over batch
            final_label_persymbol_avg_acc = final_is_correct.all(axis=2).float().mean(axis=(0,1)) # average accuracy per symbol
            metrics['step_final_label_full_acc'].append(final_label_full_acc)
            metrics['step_final_label_factor_acc'].append(final_label_factor_acc)
            metrics['step_final_label_persymbol_avg_acc'].append(final_label_persymbol_avg_acc)

            # check whether last variable is solved
            # NOTE: this only works if no padding
            assert batch.shape[0] == 1, 'batch size > 1 not supported right now'
            last_var_is_correct = (preds[:, -1] == batch_label[:, -1])[..., factors_slice]
            last_var_factor_acc = last_var_is_correct.float().mean(axis=(0))
            metrics['step_last_var_factor_acc'].append(last_var_factor_acc)
            last_var_fully_correct = last_var_is_correct.all(axis=-1).float().mean()
            metrics['step_last_var_fully_correct'].append(last_var_fully_correct)

            # calculate attention metrics
            attn_metrics = compute_attention_metrics(buffer)
            for k, v in attn_metrics.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)

            # set boolean mask of mistakes to check re-correction at next iteration
            last_mistakes = torch.logical_not(is_correct)

            if verbose:
                print('factor_acc', factor_acc)
                print('fully_correct_acc', fully_correct_acc)

    metrics['recorrection'] = torch.tensor(metrics['recorrection'])

    for k, v in metrics.items():
        if isinstance(v, list):
            metrics[k] = torch.stack(v)

    # check whether final prediction is correct (i.e., full solved)
    final_is_correct = (step_stack[-1] == batch_label)[..., factors_slice]

    final_is_correct = torch.logical_or(final_is_correct, is_pad)

    final_label_full_acc = final_is_correct.all(axis=(1,2)).float().mean().item()
    final_label_factor_acc = final_is_correct.float().mean(axis=(0,1))
    metrics['final_label_full_acc'] = final_label_full_acc
    metrics['final_label_factor_acc'] = final_label_factor_acc

    # check whether last variable in final prediction is solved
    last_var_is_correct = (step_stack[-1][:, -1] == batch_label[:, -1])[..., factors_slice]
    last_var_factor_acc = last_var_is_correct.float().mean(axis=(0,1))
    metrics['last_var_factor_acc'] = last_var_factor_acc
    last_var_fully_correct = last_var_is_correct.all(axis=-1).float().mean()
    metrics['last_var_fully_correct'] = last_var_fully_correct

    if verbose:
        print('final_label_full_acc', final_label_full_acc)
        print('final_label_factor_acc', final_label_factor_acc)

    return metrics


def calc_metrics_across_batches(litmodel, dataloader, factored_tokenizer, n_batches=None, n_steps=None, factors_slice=None, verbose=False, func_to_call=calc_batch_metrics):

    if factors_slice is None:
        factors_slice = slice(0, 6) # consider all factors

    metrics = dict(factor_acc=[], position_acc=[], fully_correct_acc=[], recorrection=[])

    for i, batch in tqdm(enumerate(dataloader)):
        if n_batches is not None and i >= n_batches:
            break

        # move batch to device
        if batch.device != litmodel.device:
            batch = batch.to(litmodel.device)

        batch_metrics = func_to_call(litmodel, batch, factored_tokenizer, n_steps=n_steps, factors_slice=factors_slice, verbose=verbose)

        for k, v in batch_metrics.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)

    metrics['final_label_full_acc'] = torch.tensor(metrics['final_label_full_acc'])

    for k, v in metrics.items():
        if isinstance(v, list):
            if isinstance(v[0], torch.Tensor):
                metrics[k] = torch.stack(v, dim=0)

    return metrics

# region prediction on sample

def print_step(inputs, preds, factored_tokenizer, label=None, stop_at_pad=True, col_width=12, factors_slice=None, return_str=False):

    if return_str:
        str_out = ''
        def print(*args, **kwargs):
            nonlocal str_out
            end = kwargs.get('end', '\n')
            str_out += ' '.join(map(str, args)) + end

    factored_input = factored_tokenizer.decode_tokens(inputs.tolist())
    factored_preds = factored_tokenizer.decode_tokens(preds.tolist())
    if label is not None:
        factored_label = factored_tokenizer.decode_tokens(label.tolist())

    if factors_slice is None:
        factors_slice = slice(0, 6) # don't ignore any factors
    factors = factored_tokenizer.factors[factors_slice]
    # factors_slice = slice(0, 4) # ignore REGRET and MODIFICATION for now

    if label is not None:
        print('INPUT -> PREDICTION | LABEL')
    else:
        print('INPUT -> PREDICTION')
    num_sections = 3 if label is not None else 2
    print('-'*(col_width+3)*len(factors)*num_sections)

    factored_input.insert(0, factors)
    factored_preds.insert(0, factors)
    if label is not None:
        factored_label.insert(0, factors)

    factored_input.insert(1, ['-'*col_width]*len(factors))
    factored_preds.insert(1, ['-'*col_width]*len(factors))
    if label is not None:
        factored_label.insert(1, ['-'*col_width]*len(factors))

    if label is None:
        for x, yhat in zip(factored_input, factored_preds):
            if stop_at_pad and x[0] == factored_tokenizer.PAD:
                break
            x, yhat = list(x)[factors_slice], list(yhat)[factors_slice]
            for idx, (xfactor, yhatfactor) in enumerate(zip(x, yhat)):
                x[idx] = x[idx].center(col_width)
                yhat[idx] = yhat[idx].center(col_width)
                if xfactor != yhatfactor:
                    x[idx] = red(x[idx])
                    yhat[idx] = green(yhat[idx])

            for xfactor in x:
                print(xfactor, end=' | ')
            print('\b| -> |', end='| ')
            for yhatfactor in yhat:
                print(yhatfactor, end=' | ')
            print()
    else:
        for x, yhat, y in zip(factored_input, factored_preds, factored_label):
            if stop_at_pad and x[0] == factored_tokenizer.PAD:
                break
            x, yhat, y = list(x)[factors_slice], list(yhat)[factors_slice], list(y)[factors_slice]

            for idx, (xfactor, yhatfactor, yfactor) in enumerate(zip(x, yhat, y)):
                # center the text within column
                x[idx] = x[idx].center(col_width)
                yhat[idx] = yhat[idx].center(col_width)
                y[idx] = y[idx].center(col_width)

                # color-code labels based in change and correctness
                if xfactor != yhatfactor:
                    x[idx] = violet(x[idx])
                    yhat[idx] = violet(yhat[idx])
                if xfactor != yhatfactor and yhatfactor != yfactor:
                    y[idx] = red(y[idx])
                elif xfactor != yhatfactor and yhatfactor == yfactor:
                    y[idx] = green(y[idx])

            for i, xfactor in enumerate(x):
                end = ' | ' if i < len(x) - 1 else ' |'
                print(xfactor, end=end)
            print('| -> |', end='| ')
            for i, yhatfactor in enumerate(yhat):
                end = ' | ' if i < len(yhat) - 1 else ' |'
                print(yhatfactor, end=end)
            print('| || |', end='| ')
            for yfactor in y:
                print(yfactor, end=' | ')
            print()

    if return_str:
        return str_out


def print_model_steps(litmodel, batch, return_str=False, sample=0, n_steps=None):

    if return_str:
        str_out = ''
        def print(*args, **kwargs):
            nonlocal str_out
            end = kwargs.get('end', '\n')
            str_out += ' '.join(map(str, args)) + end

    # process batch
    batch = batch.to(litmodel.device)
    batch = litmodel.reduce_batch(copy.deepcopy(batch))
    model_config = litmodel.model_config
    factored_tokenizer = litmodel.teacher_model.factored_tokenizer

    batch, batch_label, batch_info = batch[..., :model_config.data_dim], batch[..., model_config.data_dim:model_config.data_dim + model_config.label_dim], batch[..., model_config.data_dim + model_config.label_dim:]
    # note: batch_label is final label

    # get teacher batch
    depths = batch_info[..., 0]
    if n_steps is None:
        n_steps = depths.max().item() + 1

    batch_ext, value_avail_pos, value_to_compute_pos = litmodel.get_teacher_batch_with_depth(batch, batch_label, depths)

    # inputs = batch[0]
    inputs_stack = [batch[sample]]
    batch_ext = batch_ext[:, sample]
    # labels = batch_ext[0, 1:]

    factor_slice = slice(0, 4) # ignore REGRET and MODIFICATION for now
    print_width = (12 + 3) * 4 * 3 # (col_width+3)*len(factors)*num_sections

    fully_corrects = []
    outputs = None
    for step in range(0, n_steps):
        inputs = inputs_stack[-1]

        # logitss = litmodel.predict_step(inputs.unsqueeze(0))
        # one_step_prediction
        outputs, logitss = one_step_prediction(litmodel, batch, step, outputs=outputs, step_stack=inputs_stack)

        preds = torch.stack([logits.argmax(-1) for logits in logitss], dim=-1)[0] # argmax prediction for each factoro
        # preds = litmodel.SAE_forward(inputs.unsqueeze(0))

        if step + 1 <= len(batch_ext) - 1:
            step_label = batch_ext[step + 1]
        else:
            step_label = batch_ext[-1]

        fully_correct = (preds == step_label)[:, factor_slice].all().detach().cpu().numpy()
        fully_corrects.append(fully_correct)

        inputs_stack.append(preds)

        print('='*print_width)
        print(f'STEP: {step + 1} (Fully Correct: {fully_correct} )')
        if return_str:
            out_str = print_step(inputs=inputs, preds=preds, factored_tokenizer=factored_tokenizer, label=step_label, factors_slice=factor_slice, return_str=True)
            print(out_str)
        else:
            print_step(inputs=inputs, preds=preds, factored_tokenizer=factored_tokenizer, label=step_label, factors_slice=factor_slice, return_str=False)
        print('='*print_width)
        print()

    # print the final label
    print('='*print_width)
    print('IS FULLY CORRECT AT ALL STEPS:', all(fully_corrects))

    matches_final_label = (inputs_stack[-1] == batch_label[sample])[:, factor_slice].all().detach().cpu().numpy()
    print('MATCHES FINAL LABEL:', matches_final_label)

    if return_str:
        return str_out

# endregion

def print_sample(sample, factored_tokenizer, print_=True):

    # decode factored tokens in sample
    factored_tokens = factored_tokenizer.decode_tokens(sample.tolist())

    # translate to readable string
    sample_string = ' '.join(factored_tokenizer.de_factor_tokens(factored_tokens))
    sample_string = sample_string.replace(f' {factored_tokenizer.EQ_SEP} ', '\n') # separate equations
    sample_string = sample_string.replace(factored_tokenizer.PAD, '.') # replace padding with a dot

    # replace op names with more readable symbols
    op_dict = {'ADD': '+', 'SUB': '-', 'MUL': '*', 'DIV': '/'}
    for op_old, op_new in op_dict.items():
        sample_string = sample_string.replace(op_old, op_new)

    if print_:
        print(sample_string)
    else:
        return sample_string

# helper functions to color text (used to highlight changes in predictions)
def blue(text):
    return f'\033[34m{text}\033[0m'

def green(text):
    return f'\033[32m{text}\033[0m'

def red(text):
    return f'\033[31m{text}\033[0m'

def yellow(text):
    return f'\033[33m{text}\033[0m'

def cyan(text):
    return f'\033[36m{text}\033[0m'

def magenta(text):
    return f'\033[35m{text}\033[0m'

def purple(text):
    return f'\033[95m{text}\033[0m'

def violet(text):
    return f'\033[35m{text}\033[0m'

def orange(text):
    return f'\033[33m{text}\033[0m'


# endregion


def calc_batch_metrics_fast(litmodel, batch, factored_tokenizer, n_steps=None, factors_slice=None, verbose=False):

    litmodel.eval()
    buffer, hook_handles = attach_hooks_to_model(litmodel)

    model_config = litmodel.model_config

    if factors_slice is None:
        factors_slice = slice(0, 6) # consider all factors

    with torch.no_grad():
        # process batch
        # model_config, data_config = litmodel.model_config, litmodel.data_config
        batch = litmodel.reduce_batch(copy.deepcopy(batch))
        batch, batch_label, batch_info = batch[..., :model_config.data_dim], batch[..., model_config.data_dim:model_config.data_dim + model_config.label_dim], batch[..., model_config.data_dim + model_config.label_dim:]
        # note: batch_label is final label

        # get teacher batch
        depths = batch_info[..., 0]
        if n_steps is None:
            n_steps = depths.max().item() + 1
            print('inferred max steps:', n_steps)

        batch_ext, value_avail_pos, value_to_compute_pos = litmodel.get_teacher_batch_with_depth(batch, batch_label, depths)
        # batch_ext.shape: [n_steps, batch_size, seq_len, n_factors]

        # infer which tokens are padding, and ignore them in computing accuracy
        pad_encoding = torch.tensor(factored_tokenizer.encode_string(factored_tokenizer.PAD)[0], device=batch.device)
        is_pad = torch.eq(batch, pad_encoding).all(dim=-1)

        # get first pad for each sample
        orig_seq_len = batch.size(1)
        seq_lens = (~is_pad).sum(dim=1)
        max_seq_len = seq_lens.max()
        if verbose:
            print('max non-pad length:', max_seq_len)
            print('orig seq len:', orig_seq_len)

        is_pad = is_pad.unsqueeze(-1).repeat(1, 1, 6)
        is_pad = is_pad[..., factors_slice]

        # calc metrics up to max_seq_len in batch
        batch = batch[:, :max_seq_len]
        batch_ext = batch_ext[:, :, :max_seq_len]
        batch_label = batch_label[:, :max_seq_len]
        is_pad = is_pad[:, :max_seq_len]

        # inputs = batch[0]
        step_stack = [batch]


        metrics =  dict(factor_acc=[], position_acc=[], fully_correct_acc=[], recorrection=[], step_final_label_full_acc=[], step_final_label_factor_acc=[], step_final_label_persymbol_avg_acc=[], step_last_var_factor_acc=[], step_last_var_fully_correct=[])

        last_mistakes = None
        outputs = None
        for step in range(0, n_steps):
            if verbose:
                print('step', step)

            # one_step_prediction
            outputs, logitss = one_step_prediction(litmodel, batch, step, outputs=outputs, step_stack=step_stack)

            preds = torch.stack([logits.argmax(-1) for logits in logitss], dim=-1) # argmax prediction for each factoror. Shape: (batch_size, seq_len, n_factors)
            step_stack.append(preds)
            # print('preds', preds.shape)

            if step + 1 <= len(batch_ext) - 1:
                step_label = batch_ext[step + 1]
            else:
                step_label = batch_ext[-1]

            is_correct = (preds == step_label)[..., factors_slice]
            is_correct = torch.logical_or(is_correct, is_pad)

            # percent recorrection
            if last_mistakes is None or last_mistakes.sum() == 0:
                recorrection = torch.nan
            else:
                recorrection = torch.masked_select(is_correct, last_mistakes).float().mean().item()
            metrics['recorrection'].append(recorrection)

            factor_acc = is_correct.float().mean(axis=(0,1))
            metrics['factor_acc'].append(factor_acc)

            # accuracy by position (accounting for all factors)
            position_acc = is_correct.all(axis=-1).float().mean(axis=(0))
            # pad postion_acc with nan until orig_seq_len
            position_acc = torch.cat([position_acc, torch.full((orig_seq_len - position_acc.size(0),), fill_value=torch.nan).to(device=batch.device)])
            metrics['position_acc'].append(position_acc)

            # check full correctness (accounting for all factors and positions)
            fully_correct = is_correct.all(axis=(1,2)) # sequence accuracy
            fully_correct_acc = fully_correct.float().mean()
            metrics['fully_correct_acc'].append(fully_correct_acc)

            # persymbol average accuracy (accounting for all factors, averaged over positions)
            persymbol_avg_acc = is_correct.all(axis=2).float().mean(axis=(0,1)).mean() # average accuracy per symbol
            metrics['persymbol_avg_acc'] = persymbol_avg_acc

            # check whether final label is correct (i.e., full solved)
            final_is_correct = (preds == batch_label)[..., factors_slice]
            final_is_correct = torch.logical_or(final_is_correct, is_pad)
            final_label_full_acc = final_is_correct.all(axis=(1,2)).float().mean() # sequence accuracy
            final_label_factor_acc = final_is_correct.float().mean(axis=(0,1)) # accuracy by factor, averaged over batch
            final_label_persymbol_avg_acc = final_is_correct.all(axis=2).float().mean(axis=(0,1)) # average accuracy per symbol
            metrics['step_final_label_full_acc'].append(final_label_full_acc)
            metrics['step_final_label_factor_acc'].append(final_label_factor_acc)
            metrics['step_final_label_persymbol_avg_acc'].append(final_label_persymbol_avg_acc)

            # check whether last variable is solved (last non-padding token in each sample)
            # NOTE: this only works if no padding
            # assert batch.shape[0] == 1, 'batch size > 1 not supported right now'

            # we support batch size > 1 here
            last_non_pad_by_sample = (~is_pad[..., 0]).sum(dim=1) - 1
            # use advanced indexing to get the last non-pad token for each sample
            batch_indices = torch.arange(preds.shape[0])

            last_var_is_correct = (preds[batch_indices, last_non_pad_by_sample, :] == batch_label[batch_indices, last_non_pad_by_sample, :])[..., factors_slice] # shape: (batch_size, n_factors)
            last_var_factor_acc = last_var_is_correct.float().mean(axis=(0))
            metrics['step_last_var_factor_acc'].append(last_var_factor_acc)
            last_var_fully_correct = last_var_is_correct.all(axis=-1).float().mean()
            metrics['step_last_var_fully_correct'].append(last_var_fully_correct)

            # calculate attention metrics
            attn_metrics = compute_attention_metrics(buffer)
            for k, v in attn_metrics.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v)

            # set boolean mask of mistakes to check re-correction at next iteration
            last_mistakes = torch.logical_not(is_correct)

            if verbose:
                print('factor_acc', factor_acc)
                print('fully_correct_acc', fully_correct_acc)

    metrics['recorrection'] = torch.tensor(metrics['recorrection'])

    for k, v in metrics.items():
        if isinstance(v, list):
            metrics[k] = torch.stack(v)

    # check whether final prediction is correct (i.e., full solved)
    final_is_correct = (step_stack[-1] == batch_label)[..., factors_slice]

    final_is_correct = torch.logical_or(final_is_correct, is_pad)

    final_label_full_acc = final_is_correct.all(axis=(1,2)).float().mean().item()
    final_label_factor_acc = final_is_correct.float().mean(axis=(0,1))
    metrics['final_label_full_acc'] = final_label_full_acc
    metrics['final_label_factor_acc'] = final_label_factor_acc

    # check whether last variable in final prediction is solved
    last_var_is_correct = (step_stack[-1][:, -1] == batch_label[:, -1])[..., factors_slice]
    last_var_factor_acc = last_var_is_correct.float().mean(axis=(0,1))
    metrics['last_var_factor_acc'] = last_var_factor_acc
    last_var_fully_correct = last_var_is_correct.all(axis=-1).float().mean()
    metrics['last_var_fully_correct'] = last_var_fully_correct

    if verbose:
        print('final_label_full_acc', final_label_full_acc)
        print('final_label_factor_acc', final_label_factor_acc)

    return metrics