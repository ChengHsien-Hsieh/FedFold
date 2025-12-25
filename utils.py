import torch
import numpy as np
import copy
import math
from collections import OrderedDict

class Compressor:
    def apply_top_k(model, k_ratio):
        """
        Apply Top-K sparsification on the model params.
        :param model: The model whose params will be sparsified.
        :param k_ratio: The ratio of params to keep (K).
        """
        with torch.no_grad():
            for param in model.parameters():
                if param is not None:
                    # Flatten the parameter tensor and compute the number of values to keep
                    param_flat = param.view(-1)
                    k = int(len(param_flat) * k_ratio)
                    if k == 0:
                        continue
                    
                    _, idxs = torch.topk(param_flat.abs(), k, sorted=False)
                    compressed_param = torch.zeros_like(param_flat)
                    compressed_param[idxs] = param_flat[idxs]
                    param.copy_(compressed_param.view_as(param))
        # with torch.no_grad():
        #     for param in model.parameters():
        #         if param.grad is not None:
        #             # Flatten the gradient tensor and compute the number of values to keep
        #             grad_flat = param.grad.view(-1)
        #             k = int(len(grad_flat) * k_ratio)
        #             if k == 0:
        #                 continue
                    
        #             _, idxs = torch.topk(grad_flat.abs(), k, sorted=False)                    
        #             compressed_grad = torch.zeros_like(grad_flat)                    
        #             compressed_grad[idxs] = grad_flat[idxs]
        #             param.grad = compressed_grad.view_as(param.grad)

    def quantize(params, bits):
        assert 1 <= bits <= 32, "Bits should be between 1 and 32."
        if bits == 1:
            quantized = torch.sign(params)
            return quantized
        max_val = params.abs().max()
        qmax = 2**(bits - 1) - 1
        scale = max_val / qmax
        quantized = torch.round(params / scale).clamp(-qmax, qmax)
        dequantized = quantized * scale

        return dequantized

    def apply_svd(model_params, percent):
        svd_params = {}

        for param_name, param in model_params.items():
            if param.ndim >= 2:  # Apply SVD only to tensors with at least 2 dimensions
                if param.ndim > 2:
                    shape = param.shape
                    param_2d = param.view(shape[0], -1)  # Flatten to (out_channels, in_channels * kernel_size)
                else:
                    param_2d = param

                # Get indices of the top K largest singular values
                U, S, V = torch.svd(param_2d)
                K = int(len(S) * percent)
                topK_indices = torch.topk(S, K, largest=True).indices
                U_k = U[:, topK_indices]  
                S_k = S[topK_indices]    
                V_k = V[:, topK_indices]

                svd_params[param_name] = (U_k, S_k, V_k, shape if param.ndim > 2 else None)
            else:
                svd_params[param_name] = param  # Skip SVD for 1D tensors (like biases)
        
        return svd_params
    
    def reconstruct_svd(svd_params):
        model_params = {}

        for param_name, svd_tuple in svd_params.items():
            if isinstance(svd_tuple, tuple):  # Only reconstruct if it's a tuple (i.e., compressed)
                U_k, S_k, V_k, shape = svd_tuple
                param_reconstructed = U_k @ torch.diag(S_k) @ V_k.T
                if shape is not None:
                    param_reconstructed = param_reconstructed.view(shape)

                model_params[param_name] = param_reconstructed
            else:
                model_params[param_name] = svd_tuple
        return model_params

    
class Utils:
    # Retrieve model parameters for the specified size
    def get_model_params(idx, model_list):
        model = model_list[idx]
        return {name: param.data for name, param in model.named_parameters()}
    
    # align all model to the left upper corner, with scalar corresponding to the contribution of numbers of clients
    def accum_model(models):
        # model order in models is from largest model -> smallest model
        accum_model_params = OrderedDict()  # Accumulated parameters for aggregation
        count_params = OrderedDict()  # Keep track of how many models contribute to each parameter

        local_model_params = {}
        accum_model_params = {}
        
        for model in models:
            if not accum_model_params:
                accum_model_params = {k: v.clone() for k, v in model.items()}
                for k, v in model.items():
                    count_params[k] = torch.ones_like(v, dtype=torch.float32)
                # print(f"Initial accum_model_params: {accum_model_params['layer1.1.conv2.weight'][0][0]}")
            else:
                for k, v in model.items():
                    if accum_model_params[k].shape != v.shape:
                        diff = [accum_model_params[k].size(dim) - v.size(dim) for dim in range(v.dim())]
                        if all(d >= 0 for d in diff):
                            padding = []
                            for d in reversed(diff):
                                padding.extend((0, d))  # Padding format (pad_left, pad_right)
                            padded_v = torch.nn.functional.pad(v, padding)
                            accum_model_params[k] += padded_v
                            count_params[k] += (padded_v != 0).float() 

                        else:
                            raise ValueError(f"Cannot pad tensor {k}, model_params is larger in some dimensions")
                    else:
                        accum_model_params[k] += v.clone()
                        count_params[k] += (v != 0).float() 
                
        #scalar      
        for k in accum_model_params:
            # print(f'cont = {count_params[k]}')
            accum_model_params[k] = accum_model_params[k] / count_params[k].clamp(min=1)
        
        local_model_params = {k: v.clone() for k, v in accum_model_params.items()}
        return local_model_params
    
    # split the aggregate model, no split linear part
    def split_resnet_params(global_params, hidden_sizes, moving_spitting):
        models = [{} for _ in hidden_sizes]
        n_class = 10

        for k, concat_param in global_params.items():
            start_idx1 = 0
            start_idx2 = 0
            # print(f"Global model param shape for {k}: {concat_param.shape}")
            # Split the parameters based on each model's hidden size
            for i, hidden_size in enumerate(hidden_sizes):
                if 'layer1' in k:
                    param_size1 = hidden_size[0]
                    param_size2 = hidden_size[0]
                elif 'layer2' in k:
                    param_size1 = hidden_size[1]
                    if 'layer2.0.conv1' in k or 'layer2.0.shortcut.weight' in k:
                        param_size2 = hidden_size[0]
                    else:
                        param_size2 = hidden_size[1]
                elif 'layer3' in k:
                    param_size1 = hidden_size[2]
                    if 'layer3.0.conv1' in k or 'layer3.0.shortcut.weight' in k:
                        param_size2 = hidden_size[1]
                    else:
                        param_size2 = hidden_size[2]
                elif 'layer4' in k:
                    param_size1 = hidden_size[3]
                    if 'layer4.0.conv1' in k or 'layer4.0.shortcut.weight' in k:
                        param_size2 = hidden_size[2]
                    else:
                        param_size2 = hidden_size[3]
                elif 'conv1' in k:
                    param_size1 = hidden_size[0]
                    param_size2 = hidden_size[0]
                elif 'linear.weight' in k:
                    param_size1 = n_class
                    param_size2 = hidden_size[3]
                elif 'linear.bias' in k:
                    param_size1 = n_class
                    param_size2 = None
                else:
                    raise ValueError(f"Unknown layer key: {k}")               

                if moving_spitting:
                    if concat_param.ndim == 1:  # Biases (1D tensors)
                        models[i][k] = concat_param[:param_size1].clone()
                    else:  # 2D+ tensors
                        models[i][k] = concat_param[start_idx1:start_idx1 + param_size1, start_idx2:start_idx2+param_size2, ...].clone()
                    if 'linear.bias' in k:
                        continue
                    elif k=='conv1.weight':
                        start_idx1 += param_size1
                    elif 'linear.weight' in k:
                        start_idx2 += param_size2
                    else:
                        start_idx2 += param_size2
                        start_idx1 += param_size1
                else:
                    if concat_param.ndim == 1:  # Biases (1D tensors)
                        models[i][k] = concat_param[:param_size1].clone()
                    else:  # 2D+ tensors
                        models[i][k] = concat_param[start_idx1:start_idx1 + param_size1, :param_size2, ...].clone()
                
                # print(f"Split model {i} param shape for {k}: {models[i][k].shape}")
                
        return models
    
    def split_cnn_params(global_params, hidden_sizes, moving_splitting):
        models = [{} for _ in hidden_sizes]
        n_class = 10
        for k, concat_param in global_params.items():
            start_idx1 = 0
            start_idx2 = 0
            for i, hidden_size in enumerate(hidden_sizes):
                # print(f"Global model param shape for {k}: {concat_param.shape}")
                if '0' in k:
                    param_size1 = hidden_size[0]
                    if 'bias' in k:
                        param_size2 = None
                    else:
                        param_size2 = 3
                elif '13.weight' in k:
                    param_size1 = n_class
                    param_size2 = hidden_size[3]
                elif '13.bias' in k:
                    param_size1 = n_class
                    param_size2 = None
                elif '3' in k:
                    param_size1 = hidden_size[1]
                    if 'bias' in k:
                        param_size2 = None
                    else:
                        param_size2 = hidden_size[0]
                elif '6' in k:
                    param_size1 = hidden_size[2]
                    if 'bias' in k:
                        param_size2 = None
                    else:
                        param_size2 = hidden_size[1]
                elif '9' in k:
                    param_size1 = hidden_size[3]
                    if 'bias' in k:
                        param_size2 = None
                    else:
                        param_size2 = hidden_size[2]
                
                if moving_splitting:
                    if concat_param.ndim == 1:  # Biases (1D tensors)
                        models[i][k] = concat_param[:param_size1].clone()
                    else:  # 2D+ tensors
                        models[i][k] = concat_param[start_idx1:start_idx1 + param_size1, start_idx2:start_idx2+param_size2, ...].clone()
                    if k=='blocks.13.bias':
                        continue
                    elif k=='blocks.0.weight' or concat_param.ndim == 1:
                        start_idx1 += param_size1
                    elif 'blocks.13.weight' in k:
                        start_idx2 += param_size2
                    else:
                        start_idx2 += param_size2
                        start_idx1 += param_size1
                else:
                    if concat_param.ndim == 1:  # Biases (1D tensors)
                        models[i][k] = concat_param[:param_size1].clone()
                    else:  # 2D+ tensors
                        models[i][k] = concat_param[:param_size1, :param_size2, ...].clone()
                # print(f"Split model {i} param shape for {k}: {models[i][k].shape}")
        return models

    def split_model(local_model, split_size, model_type, moving_splitting):
        split_models = []
        
        hidden_size = Utils.get_hidden_size(split_size)
            
        if model_type == 'ResNet':
            split_models.extend(Utils.split_resnet_params(local_model, hidden_size, moving_splitting))
        elif model_type =='Conv':
            split_models.extend(Utils.split_cnn_params(local_model, hidden_size, moving_splitting))
        # print(len(models))
        return split_models

    # This function is very useless
    # it can only be used when all the models you want to combine are in the same size
    def combine(models):
        local_model_params = {}
        accum_model_params = {}
        for idx, model_params in enumerate(models):
            if not accum_model_params:
                accum_model_params = {k: v.clone()  for k, v in model_params.items()}
            else:
                for k, v in model_params.items():
                    accum_model_params[k] += v.clone()
        local_model_params = {k: v.clone() for k, v in accum_model_params.items()}
        # print(f"Local: {local_model_params['layer1.1.conv2.weight'][0][0][0][0]}")
        return local_model_params
    
    def concat_model(model, hidden_sizes, model_type):
        if model_type == 'ResNet':
            return Utils.concat_resnet(model, hidden_sizes)
        elif model_type == 'Conv':
            return Utils.concat_cnn(model, hidden_sizes)

    def concat_resnet(model, hidden_sizes):    
        concatenated_params = []
        model_tmp = copy.deepcopy(model)
        for hidden_size in hidden_sizes:
            if hidden_size[0] == 4: 
                concatenated_params.append(model_tmp)
            else:         
                concat_model = copy.deepcopy(model_tmp)
                for _ in range(int(math.log(hidden_size[0],2)-2)): #-2
                    model = copy.deepcopy(concat_model)                                  
                    for k, v in model.items():                                        
                        if 'linear.bias' in k:
                            continue
                        elif 'linear' in k:
                            concat_model[k] = torch.cat((concat_model[k], v), dim=1)
                        elif k=='conv1.weight':
                            concat_model[k] = torch.cat((concat_model[k], v), dim=0)
                        else:
                            concat_model[k] = torch.cat((concat_model[k], v), dim=0)
                            diff = [concat_model[k].size(dim) - v.size(dim) for dim in range(v.dim())]
                            if any(d > 0 for d in diff):
                                padding = []
                                for d in reversed(diff):
                                    padding.extend((0, d))
                                padded_v = torch.nn.functional.pad(v, padding)
                            concat_model[k] = torch.cat((concat_model[k], padded_v), dim=1)
                            # print(f"Shape of concat_model[{k}]: {concat_model[k].shape}")                                                                                                                
                concatenated_params.append(concat_model)
        # print(len(concatenated_params))
        return concatenated_params
    
    def concat_cnn(model, hidden_sizes):    
        concatenated_params = []
        model_tmp = copy.deepcopy(model)
        for hidden_size in hidden_sizes:
            if hidden_size[0] == 4:
                concatenated_params.append(model)
            else:         
                concat_model = copy.deepcopy(model_tmp)
                for _ in range(int(math.log(hidden_size[0],2)-2)):
                    model = copy.deepcopy(concat_model)                                  
                    for k, v in model.items():                                        
                        if 'blocks.13.bias' in k:
                            continue
                        elif 'blocks.13.weight' in k:
                            concat_model[k] = torch.cat((concat_model[k], v), dim=1)
                        elif k=='blocks.0.weight' or v.ndim==1 :
                            concat_model[k] = torch.cat((concat_model[k], v), dim=0)
                        else:
                            concat_model[k] = torch.cat((concat_model[k], v), dim=0)
                            diff = [concat_model[k].size(dim) - v.size(dim) for dim in range(v.dim())]
                            if any(d > 0 for d in diff):
                                padding = []
                                for d in reversed(diff):
                                    padding.extend((0, d))
                                padded_v = torch.nn.functional.pad(v, padding)
                            concat_model[k] = torch.cat((concat_model[k], padded_v), dim=1)
                            # print(f"Shape of concat_model[{k}]: {concat_model[k].shape}")                                                                                                                
                concatenated_params.append(concat_model)
        # print(len(concatenated_params))
        return concatenated_params


    def get_hidden_size(split_size):
        
        if split_size == 1:
            hidden_size = [
                [4, 8, 16, 32],
                [4, 8, 16, 32],
                [4, 8, 16, 32],
                [4, 8, 16, 32],
                [4, 8, 16, 32],
                [4, 8, 16, 32],
                [4, 8, 16, 32],
                [4, 8, 16, 32],
            ]
            return hidden_size
        elif split_size==2:
            hidden_size = [
                [8,16,32,64],
                [8,16,32,64],
                [8,16,32,64],
                [8,16,32,64],
            ]
            return hidden_size
        elif split_size==4:
            hidden_size = [
                [16,32,64,128],
                [16,32,64,128],
            ]
            return hidden_size
        else:
            hidden_size = [
                [4, 8, 16, 32],
                [4, 8, 16, 32],
                [8, 16, 32, 64],
                [16, 32, 64, 128],
            ]
            return hidden_size

