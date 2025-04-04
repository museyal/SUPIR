import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.nn.functional import interpolate
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
import concurrent.futures
import time


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
    # Instantiate the model structure first (potentially using meta tensors)
    print("Instantiating model structure...")
    model = instantiate_from_config(config.model)
    print(f'Loaded model config from [{config_path}]')

    if load_weights:
        print("Starting concurrent weight loading...")
        start_time = time.time()
        checkpoints_to_load = []
        
        # Determine which checkpoints are needed
        if config.get('SDXL_CKPT'): # Use .get for safer access
            checkpoints_to_load.append(config.SDXL_CKPT)
            print(f"  - Queued SDXL: {os.path.basename(config.SDXL_CKPT)}")
            
        # Deprecated single SUPIR_CKPT, prefer F/Q specific
        # if config.get('SUPIR_CKPT'): 
        #     checkpoints_to_load.append(config.SUPIR_CKPT)
        #     print(f"  - Queued SUPIR (generic): {os.path.basename(config.SUPIR_CKPT)}")

        if SUPIR_sign:
            assert SUPIR_sign in ['F', 'Q']
            ckpt_key = f'SUPIR_CKPT_{SUPIR_sign}'
            if config.get(ckpt_key):
                checkpoints_to_load.append(config.get(ckpt_key))
                print(f"  - Queued SUPIR-{SUPIR_sign}: {os.path.basename(config.get(ckpt_key))}")
        
        # Helper function for loading a single checkpoint
        def load_single_checkpoint(path):
            print(f"    Starting load for: {os.path.basename(path)}")
            load_start = time.time()
            sd = load_state_dict(path)
            load_end = time.time()
            print(f"    Finished load for: {os.path.basename(path)} in {load_end - load_start:.2f}s")
            return sd

        # Load checkpoints concurrently
        loaded_state_dicts = []
        # Use a number of workers appropriate for concurrent I/O, 
        # limited by the number of checkpoints to avoid creating unnecessary threads.
        max_workers = min(len(checkpoints_to_load), 4) # Example: Limit to 4 workers or fewer
        if max_workers > 0:
             with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {executor.submit(load_single_checkpoint, path): path for path in checkpoints_to_load}
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        state_dict = future.result()
                        loaded_state_dicts.append(state_dict)
                    except Exception as exc:
                        print(f'ERROR: {os.path.basename(path)} generated an exception during loading: {exc}')
        
        # Merge the state dicts (later dicts overwrite earlier ones if keys conflict)
        print("Merging loaded state dictionaries...")
        merged_state_dict = {}
        for sd in loaded_state_dicts:
            merged_state_dict.update(sd)
        print("State dictionaries merged.")

        # Apply the merged state dict to the model structure
        print("Applying merged state dictionary to model structure...")
        apply_start = time.time()
        # Using strict=False as before, assuming partial loading is intended
        model.load_state_dict(merged_state_dict, strict=False) 
        apply_end = time.time()
        print(f"Merged state dictionary applied in {apply_end - apply_start:.2f}s.")
        
        total_load_time = time.time() - start_time
        print(f"Concurrent weight loading and application finished in {total_load_time:.2f} seconds.")

    if load_default_setting:
        default_setting = config.get('default_setting', {}) # Use .get for safety
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
    start_time = time.time()
    print(f"Creating empty model structure from {config_path} (may use meta tensors internally)")
    
    # Load configuration
    config = OmegaConf.load(config_path)
    model_config = config.model
    
    # Split model initialization into parallelizable components
    # This is model-specific, but most diffusion models have similar components
    def init_unet():
        try:
            # Try to create just the UNet component
            if 'unet_config' in model_config:
                from sgm.util import instantiate_from_config
                unet = instantiate_from_config(model_config.unet_config)
                print(f"UNet initialized in {time.time() - start_time:.2f} seconds")
                return ('unet', unet)
        except Exception as e:
            print(f"Error initializing UNet: {e}")
        return None
    
    def init_first_stage():
        try:
            # Try to create just the first stage model (VAE)
            if 'first_stage_config' in model_config:
                from sgm.util import instantiate_from_config
                first_stage = instantiate_from_config(model_config.first_stage_config)
                print(f"First stage model initialized in {time.time() - start_time:.2f} seconds")
                return ('first_stage_model', first_stage)
        except Exception as e:
            print(f"Error initializing first stage model: {e}")
        return None
    
    def init_cond_stage():
        try:
            # Try to create just the conditioning model
            if 'cond_stage_config' in model_config:
                from sgm.util import instantiate_from_config
                cond_stage = instantiate_from_config(model_config.cond_stage_config)
                print(f"Conditioning stage model initialized in {time.time() - start_time:.2f} seconds")
                return ('cond_stage_model', cond_stage)
        except Exception as e:
            print(f"Error initializing conditioning stage model: {e}")
        return None
    
    # Try parallel initialization first
    try:
        # Create base model
        model = create_SUPIR_model(config_path, SUPIR_sign=SUPIR_sign, load_weights=False)
        parallel_init_start = time.time()
        
        # Check if components can be initialized separately
        components = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            component_futures = [
                executor.submit(init_unet),
                executor.submit(init_first_stage),
                executor.submit(init_cond_stage)
            ]
            
            for future in concurrent.futures.as_completed(component_futures):
                result = future.result()
                if result:
                    components.append(result)
        
        # If we have at least one component initialized separately, use them
        if components:
            print(f"Parallel component initialization completed in {time.time() - parallel_init_start:.2f} seconds")
            for component_name, component in components:
                try:
                    # Replace the component in the model
                    setattr(model, component_name, component)
                    print(f"Replaced {component_name} with separately initialized component")
                except Exception as e:
                    print(f"Error replacing {component_name}: {e}")
        
        return model
    except Exception as e:
        print(f"Parallel initialization failed: {e}, falling back to standard method")
        # Fall back to standard method
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
