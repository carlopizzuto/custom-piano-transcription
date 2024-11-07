import torch
import torch.nn as nn
import torch.nn.functional as F

def safe_relu(x):
    """Non-in-place version of ReLU"""
    return F.relu(x.detach().clone(), inplace=False)

def safe_transpose(x, dim1, dim2):
    """Safe transpose operation that creates a new tensor"""
    return x.detach().clone().transpose(dim1, dim2)

class SafeReLU(nn.Module):
    """Drop-in replacement for nn.ReLU that ensures non-in-place operation"""
    def forward(self, x):
        return safe_relu(x)

def replace_relu_with_non_inplace(model):
    """Replace all in-place ReLU operations with non-in-place versions"""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, SafeReLU())
        elif isinstance(module, nn.BatchNorm1d):
            # Ensure BatchNorm doesn't modify tensors in-place
            new_bn = nn.BatchNorm1d(module.num_features, 
                                  eps=module.eps, 
                                  momentum=module.momentum,
                                  affine=module.affine,
                                  track_running_stats=module.track_running_stats)
            new_bn.load_state_dict(module.state_dict())
            setattr(model, name, new_bn)
        else:
            replace_relu_with_non_inplace(module)
    return model

def prepare_model_for_training(model):
    """Prepare model for training by replacing in-place operations"""
    model = replace_relu_with_non_inplace(model)
    return model