import copy
import torch
from utils import Compressor, Utils

class Federation:
    def __init__(self, global_params): 
        self.global_params = []
        self.accum_global_params = []
        self.cnt = []

        for i in range(len(global_params)): # 8 small + 2 large
            # self.global_params.append({k: copy.deepcopy(v) for k, v in global_params[i].items()})
            self.global_params.append({k: v.detach().clone() for k, v in global_params[i].items()})
            self.accum_global_params.append({k: torch.zeros_like(v) for k, v in global_params[i].items()})
            self.cnt.append(0)
        
    def download(self, local_params, idx):
        for k, v in self.global_params[idx].items():
            # local_params[k] = copy.deepcopy(v)
            local_params[k] = v.detach().clone()

    def upload(self, local_params, idx, need_split, need_expand):
        model_type = 'ResNet'
        hidden_size = [
            [4, 8, 16, 32],
            [4, 8, 16, 32],
            [8, 16, 32, 64],
            [16, 32, 64, 128],
            [32, 64, 128, 256], 
        ]
        if need_split:           
            split_models = Utils.split_model(local_params, 8,model_type, 0) # :-1          
            for i in range(len(split_models)):
                self.cnt[i]+=1
                for k, v in split_models[i].items():
                    # print(f"split model[{idx}] has parameter {k} has shape {v.shape}")
                    # print(f"accum model[{idx}] has parameter {k} has shape {self.accum_global_params[idx][k].shape}")
                    self.accum_global_params[i][k] += v
        if need_expand:
            # print(hidden_size[3:]) 
            concat_models = Utils.concat_model(local_params, hidden_size[1:], model_type) #1:
            cnt=1 #1
            for model in concat_models:
                for k, v in model.items():
                    self.accum_global_params[cnt][k] += v
                self.cnt[cnt] += 1
                cnt+=1
        for k, v in local_params.items():
            self.accum_global_params[idx][k] += v
            # print(v)
        self.cnt[idx] += 1

    def aggregate(self):
        for idx in range(len(self.global_params)):
            if self.cnt[idx] != 0:
                print(f"index {idx} has {self.cnt[idx]} elements")
                for k, v in self.accum_global_params[idx].items():
                    # self.global_params[idx][k] = copy.deepcopy(v) / self.cnt[idx]
                    self.global_params[idx][k] = v.detach().clone() / self.cnt[idx]
                    self.accum_global_params[idx][k] = torch.zeros_like(v)
                self.cnt[idx] = 0
