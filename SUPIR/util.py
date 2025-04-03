import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.nn.functional import interpolate
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location), weights_only=True))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model


def create_SUPIR_model(config_path, SUPIR_sign=None, load_default_setting=False, load_weights=True):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    print(f'Loaded model config from [{config_path}]')
    
    if load_weights:
        if config.SDXL_CKPT is not None:
            model.load_state_dict(load_state_dict(config.SDXL_CKPT), strict=False)
        if config.SUPIR_CKPT is not None:
            model.load_state_dict(load_state_dict(config.SUPIR_CKPT), strict=False)
        if SUPIR_sign is not None:
            assert SUPIR_sign in ['F', 'Q']
            if SUPIR_sign == 'F':
                model.load_state_dict(load_state_dict(config.SUPIR_CKPT_F), strict=False)
            elif SUPIR_sign == 'Q':
                model.load_state_dict(load_state_dict(config.SUPIR_CKPT_Q), strict=False)
    
    if load_default_setting:
        default_setting = config.default_setting
        return model, default_setting
    return model

def create_empty_SUPIR_model(config_path, SUPIR_sign=None):
    """Create an empty SUPIR model structure without loading weights.
    This helps avoid the meta tensor error during model instantiation.
    The model may be instantiated with meta tensors initially, which have no memory footprint.
    
    Args:
        config_path: Path to the SUPIR model config
        SUPIR_sign: 'F' or 'Q' indicating which model variant to use
        
    Returns:
        An instantiated but empty model structure
    """
    print(f"Creating empty model structure from {config_path} (may use meta tensors internally)")
    return create_SUPIR_model(config_path, SUPIR_sign=SUPIR_sign, load_weights=False)

def apply_weights_to_model(model, state_dict, strict=False):
    """Apply weights to a model using zero-copy techniques when possible.
    
    Args:
        model: The model to load weights into
        state_dict: Dictionary of parameter tensors
        strict: Whether to strictly enforce that the keys in state_dict match the keys in the model
        
    Returns:
        The model with weights applied
    """
    # First prepare the state dict to ensure compatibility with meta tensors
    prepared_dict = prepare_state_dict_for_meta_tensors(state_dict)
    
    try:
        # First try the standard way
        incompatible_keys = model.load_state_dict(prepared_dict, strict=strict)
        if incompatible_keys.missing_keys:
            print(f"Missing keys: {len(incompatible_keys.missing_keys)}")
        if incompatible_keys.unexpected_keys:
            print(f"Unexpected keys: {len(incompatible_keys.unexpected_keys)}")
        return model
    except RuntimeError as e:
        # If standard approach fails, try parameter by parameter
        print(f"Standard loading failed: {e}. Trying parameter-by-parameter loading.")
        
        # Keep track of applied parameters
        applied_params = 0
        total_params = 0
        
        # Iterate over all named parameters
        for name, param in model.named_parameters():
            total_params += 1
            if name in prepared_dict:
                # Handle potential device mismatch
                src_param = prepared_dict[name]
                if param.shape == src_param.shape:
                    # Check if parameter is on meta device
                    if param.device.type == 'meta':
                        # Create empty tensor first
                        if hasattr(param, 'to_empty'):
                            param = param.to_empty(device='cpu')
                        else:
                            # Manually create empty tensor
                            param.data = torch.empty(param.shape, device='cpu')
                    
                    # Zero-copy when possible by using storage offset
                    try:
                        param.data = src_param.data
                        applied_params += 1
                    except:
                        # Fallback to copy
                        param.data.copy_(src_param.data)
                        applied_params += 1
                else:
                    print(f"Shape mismatch for {name}: {param.shape} vs {src_param.shape}")
        
        # Also handle buffers (non-parameter tensors)
        for name, buffer in model.named_buffers():
            total_params += 1
            if name in prepared_dict:
                src_buffer = prepared_dict[name]
                if buffer.shape == src_buffer.shape:
                    # Check if buffer is on meta device
                    if buffer.device.type == 'meta':
                        # Create empty tensor first
                        buffer = torch.empty(buffer.shape, device='cpu')
                    
                    try:
                        buffer.copy_(src_buffer)
                        applied_params += 1
                    except:
                        print(f"Failed to copy buffer: {name}")
        
        print(f"Applied {applied_params}/{total_params} parameters")
        return model

def load_QF_ckpt(config_path):
    config = OmegaConf.load(config_path)
    ckpt_F = torch.load(config.SUPIR_CKPT_F, map_location='cpu', weights_only=True)
    ckpt_Q = torch.load(config.SUPIR_CKPT_Q, map_location='cpu', weights_only=True)
    return ckpt_Q, ckpt_F

def load_Q_ckpt(config_path):
    config = OmegaConf.load(config_path)
    _, extension = os.path.splitext(config.SUPIR_CKPT_Q)
    
    if extension.lower() == ".safetensors":
        # For safetensors format (pruned models)
        import safetensors.torch
        return safetensors.torch.load_file(config.SUPIR_CKPT_Q, device='cpu')
    else:
        return torch.load(config.SUPIR_CKPT_Q, map_location='cpu', weights_only=True)

def load_F_ckpt(config_path):
    config = OmegaConf.load(config_path)
    _, extension = os.path.splitext(config.SUPIR_CKPT_F)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        return safetensors.torch.load_file(config.SUPIR_CKPT_F, device='cpu')
    else:
        return torch.load(config.SUPIR_CKPT_F, map_location='cpu', weights_only=True)

def load_ckpt(config_path, supir_sign: str = 'Q', pruned: bool = False):
    config = OmegaConf.load(config_path)
    
    if supir_sign not in ['Q', 'F']:
        print(f'Invalid supir_sign: {supir_sign}. Valid values are: Q, F. Using Q model instead.')
        supir_sign = 'Q'
    
    ckpt_key = f'SUPIR_CKPT_{supir_sign}'
    ckpt_path = config[ckpt_key]
    
    # Check file extension to determine loading method
    _, extension = os.path.splitext(ckpt_path)
    
    try:
        if extension.lower() == ".safetensors":
            import safetensors.torch
            return safetensors.torch.load_file(ckpt_path, device='cpu')
        else:
            # For .ckpt format (full models)
            return torch.load(ckpt_path, map_location='cpu', weights_only=not pruned)
    except Exception as e:
        print(f'Error loading checkpoint: {e}. Trying alternative loading method.')
        if extension.lower() == ".safetensors":
            # Try alternative loading for safetensors
            import safetensors.torch
            return safetensors.torch.load_file(ckpt_path, device='cpu')
        else:
            # Try alternative loading for .ckpt
            return torch.load(ckpt_path, map_location='cpu', weights_only=pruned)


def PIL2Tensor(img, upsacle=1, min_size=1024, fix_resize=None):
    '''
    PIL.Image -> Tensor[C, H, W], RGB, [-1, 1]
    '''
    # size
    w, h = img.size
    w *= upsacle
    h *= upsacle
    w0, h0 = round(w), round(h)
    if min(w, h) < min_size:
        _upsacle = min_size / min(w, h)
        w *= _upsacle
        h *= _upsacle
    if fix_resize is not None:
        _upsacle = fix_resize / min(w, h)
        w *= _upsacle
        h *= _upsacle
        w0, h0 = round(w), round(h)
    w = int(np.round(w / 64.0)) * 64
    h = int(np.round(h / 64.0)) * 64
    x = img.resize((w, h), Image.BICUBIC)
    x = np.array(x).round().clip(0, 255).astype(np.uint8)
    x = x / 255 * 2 - 1
    x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
    return x, h0, w0


def Tensor2PIL(x, h0, w0):
    '''
    Tensor[C, H, W], RGB, [-1, 1] -> PIL.Image
    '''
    x = x.unsqueeze(0)
    x = interpolate(x, size=(h0, w0), mode='bicubic')
    x = (x.squeeze(0).permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def upscale_image(input_image, upscale, min_size=None, unit_resolution=64):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    H *= upscale
    W *= upscale
    if min_size is not None:
        if min(H, W) < min_size:
            _upsacle = min_size / min(W, H)
            W *= _upsacle
            H *= _upsacle
    H = int(np.round(H / unit_resolution)) * unit_resolution
    W = int(np.round(W / unit_resolution)) * unit_resolution
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if upscale > 1 else cv2.INTER_AREA)
    img = img.round().clip(0, 255).astype(np.uint8)
    return img


def fix_resize(input_image, size=512, unit_resolution=64):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    upscale = size / min(H, W)
    H *= upscale
    W *= upscale
    H = int(np.round(H / unit_resolution)) * unit_resolution
    W = int(np.round(W / unit_resolution)) * unit_resolution
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if upscale > 1 else cv2.INTER_AREA)
    img = img.round().clip(0, 255).astype(np.uint8)
    return img



def Numpy2Tensor(img):
    '''
    np.array[H, w, C] [0, 255] -> Tensor[C, H, W], RGB, [-1, 1]
    '''
    # size
    img = np.array(img) / 255 * 2 - 1
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    return img


def Tensor2Numpy(x, h0=None, w0=None):
    '''
    Tensor[C, H, W], RGB, [-1, 1] -> PIL.Image
    '''
    if h0 is not None and w0 is not None:
        x = x.unsqueeze(0)
        x = interpolate(x, size=(h0, w0), mode='bicubic')
        x = x.squeeze(0)
    x = (x.permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    return x


def convert_dtype(dtype_str):
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError

def prepare_state_dict_for_meta_tensors(state_dict):
    """Prepares a state_dict for use with meta tensors by removing device information.
    
    This ensures that the state_dict can be safely applied to models with meta tensors.
    
    Args:
        state_dict: The state dictionary to prepare
        
    Returns:
        The prepared state dictionary
    """
    prepared_dict = {}
    for k, v in state_dict.items():
        if hasattr(v, 'cpu'):
            # Detach and convert to CPU
            prepared_dict[k] = v.detach().cpu()
        else:
            prepared_dict[k] = v
    return prepared_dict
