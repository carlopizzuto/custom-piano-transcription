import torch
import torch.nn as nn

def replace_relu_with_non_inplace(model):
    """Replace all in-place ReLU operations with non-in-place versions"""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, nn.ReLU(inplace=False))
        else:
            replace_relu_with_non_inplace(module)
    return model

def prepare_model_for_training(model):
    """Prepare model for training by replacing in-place operations"""
    model = replace_relu_with_non_inplace(model)
    return model 