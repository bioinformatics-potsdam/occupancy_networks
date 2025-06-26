# Modified im2mesh/config.py
import yaml
from torchvision import transforms
from im2mesh import data
# Modified: removed dmc
from im2mesh import onet, r2n2, psgn, pix2mesh
from im2mesh import preprocess


method_dict = {
    # 'onet': onet, # Commented out to break circular importdependency
    'r2n2': r2n2,
    'psgn': psgn,
    'pix2mesh': pix2mesh,
    # 'dmc': dmc, # Commented out as dmc is disabled
}


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        # Reverted: Use plain yaml.load(f) for PyYAML 3.13 compatibility
        # Note: This is potentially unsafe if loading untrusted YAML files.
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            # Reverted: Use plain yaml.load(f) for PyYAML 3.13 compatibility
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg



def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        # Removed this check as it prevents adding new keys:
        # if k not in dict1:
        #      dict1[k] = dict()
        if isinstance(v, dict) and k in dict1 and isinstance(dict1[k], dict):
             update_recursive(dict1[k], v)
        else:
            # Overwrite or add the key/value
            dict1[k] = v


# Models
def get_model(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    # Deferred import to break circular dependency still needed here or globally
    from im2mesh import onet # Ensure import is present

    method = cfg['method']

    # --- Added Special Handling for 'onet' ---
    if method == 'onet':
        model = onet.config.get_model(cfg, device=device, dataset=dataset)
    # --- Fallback to method_dict for other methods ---
    else:
        # This might raise KeyError if method is 'dmc' (since it's also commented out)
        # but ONet demo specifically uses 'onet'
        model = method_dict[method].config.get_model(
            cfg, device=device, dataset=dataset)
    return model



# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    # Import onet module here if needed
    from im2mesh import onet

    method = cfg['method']

    # --- Added Special Handling for 'onet' ---
    if method == 'onet':
        trainer = onet.config.get_trainer(model, optimizer, cfg, device)
    # --- Fallback to method_dict for other methods ---
    else:
        trainer = method_dict[method].config.get_trainer(
            model, optimizer, cfg, device)
    return trainer



# Generator for final mesh extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    # Import onet module here if needed
    from im2mesh import onet

    method = cfg['method']

    # --- Added Special Handling for 'onet' ---
    if method == 'onet':
        generator = onet.config.get_generator(model, cfg, device)
    # --- Fallback to method_dict for other methods ---
    else:
        # This might still raise KeyError for 'dmc', but handles 'onet'
        generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator



# Datasets
def get_dataset(mode, cfg, return_idx=False, return_category=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used (this argument seems unused here?)
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    # Import onet module here if needed
    from im2mesh import onet

    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        # --- Added Special Handling for 'onet' for get_data_fields ---
        if method == 'onet':
            fields = onet.config.get_data_fields(mode, cfg)
        else:
            # Fallback to method_dict for other methods
            fields = method_dict[method].config.get_data_fields(mode, cfg)
        # --- End Special Handling ---

        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        if return_category:
            fields['category'] = data.CategoryField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
        )
    elif dataset_type == 'kitti':
        dataset = data.KittiDataset(
            dataset_folder, img_size=cfg['data']['img_size'],
            return_idx=return_idx
        )
    elif dataset_type == 'online_products':
        dataset = data.OnlineProductDataset(
            dataset_folder, img_size=cfg['data']['img_size'],
            classes=cfg['data']['classes'],
            max_number_imgs=cfg['generation']['max_number_imgs'],
            return_idx=return_idx, return_category=return_category
        )
    elif dataset_type == 'images':
        dataset = data.ImageDataset(
            dataset_folder, img_size=cfg['data']['img_size'],
            return_idx=return_idx,
        )
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset


def get_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']

    # Simplified logic slightly from original for clarity
    if input_type is None:
        return None # Explicit return None

    with_transforms = cfg['data'].get('with_transforms', False) # Use .get for safety

    if input_type == 'img':
        img_augment = cfg['data'].get('img_augment', False) # Use .get
        img_size = cfg['data']['img_size'] # Assume required

        if mode == 'train' and img_augment:
            resize_op = transforms.RandomResizedCrop(
                 img_size, scale=(0.75, 1.0), ratio=(1.0, 1.0)) # Corrected arguments
        else:
            resize_op = transforms.Resize((img_size, img_size)) # Ensure tuple for Resize

        transform = transforms.Compose([
            resize_op, transforms.ToTensor(),
        ])

        with_camera = cfg['data'].get('img_with_camera', False) # Use .get
        random_view = (mode == 'train') # Simpler boolean assignment

        inputs_field = data.ImagesField(
            cfg['data']['img_folder'], # Assume required
            transform,
            with_camera=with_camera,
            random_view=random_view
        )
    elif input_type == 'pointcloud':
        pointcloud_n = cfg['data'].get('pointcloud_n', None) # Use .get
        pointcloud_noise = cfg['data'].get('pointcloud_noise', 0.0) # Use .get

        transform_list = []
        if pointcloud_n is not None:
             transform_list.append(data.SubsamplePointcloud(pointcloud_n))
        if pointcloud_noise > 0.0:
            transform_list.append(data.PointcloudNoise(pointcloud_noise))

        transform = transforms.Compose(transform_list) if transform_list else None

        inputs_field = data.PointCloudField(
            cfg['data']['pointcloud_file'], # Assume required
            transform=transform, # Pass potentially None transform
            with_transforms=with_transforms
        )
    elif input_type == 'voxels':
        inputs_field = data.VoxelsField(
            cfg['data']['voxels_file'] # Assume required
        )
    elif input_type == 'idx':
        inputs_field = data.IndexField()
    else:
        raise ValueError('Invalid input type (%s)' % input_type)

    return inputs_field


def get_preprocessor(cfg, dataset=None, device=None):
    ''' Returns preprocessor instance.

    Args:
        cfg (dict): config dictionary
        dataset (dataset): dataset
        device (device): pytorch device
    '''
    # Use .get for safer access to dictionary keys
    p_cfg = cfg.get('preprocessor', {})
    p_type = p_cfg.get('type', None)
    cfg_path = p_cfg.get('config')
    model_file = p_cfg.get('model_file')

    if p_type == 'psgn':
        preprocessor = preprocess.PSGNPreprocessor(
            cfg_path=cfg_path,
            # Use .get with default for pointcloud_n as well
            pointcloud_n=cfg.get('data', {}).get('pointcloud_n'),
            dataset=dataset,
            device=device,
            model_file=model_file,
        )
    elif p_type is None:
        preprocessor = None
    else:
        raise ValueError('Invalid Preprocessor %s' % p_type)

    return preprocessor

