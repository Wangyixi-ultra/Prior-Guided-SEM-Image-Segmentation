import os
import sys
import torch
import json
import csv
import contextlib
from typing import Tuple

# Add U-Mamba/umamba to path
# Use absolute path to ensure imports work correctly from anywhere
sys.path.insert(0, os.path.abspath('U-Mamba/umamba'))

try:
    import thop
except ImportError:
    thop = None
    print("Warning: thop not installed. FLOPs will be 0.")

from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isdir
import nnunetv2

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def calculate_complexity(model, input_size):
    # Accurate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    
    macs = 0
    if thop is not None:
        device = next(model.parameters()).device
        try:
            dummy_input = torch.randn(input_size).to(device)
            # thop requires model to be on same device as input
            with suppress_stdout():
                # verbose=False to reduce noise
                # macs might be returned as a tuple or single value depending on version, 
                # but typically thop.profile returns (macs, params)
                macs, _ = thop.profile(model, inputs=(dummy_input,), verbose=False)
        except Exception as e:
            # sys.stderr.write(f"Warning: FLOPs calculation failed: {e}\n")
            pass
            
    return macs, total_params

def build_network(trainer_class, plans_manager, dataset_json, configuration_manager, num_input_channels):
    # Logic to build network
    # 1. Try static method 'build_network_architecture'
    build_method = None
    
    # Check if build_network_architecture is static in the class or its MRO
    for cls in [trainer_class] + list(trainer_class.__mro__):
        if 'build_network_architecture' in cls.__dict__:
            attr = cls.__dict__['build_network_architecture']
            if isinstance(attr, staticmethod):
                build_method = getattr(cls, 'build_network_architecture')
                break
    
    if build_method:
        return build_method(plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision=False)

    # 2. Instantiate and use initialize_network
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    try:
        # Standard nnUNetTrainer signature
        trainer = trainer_class(plans_manager.plans, configuration_manager.configuration_name, 0, dataset_json, unpack_dataset=False, device=device)
        trainer.initialize_network()
        return trainer.network
    except Exception as e:
        print(f"Instantiation failed for {trainer_class.__name__}: {e}")
        return None

def process_dataset(dataset_path):
    print(f"\nProcessing {dataset_path}")
    print(f"{'Trainer':<60} | {'Config':<10} | {'Params (M)':<10} | {'FLOPs (G)':<10}")
    print("-" * 100)
    
    results = []
    
    if not os.path.isdir(dataset_path):
        print(f"Directory not found: {dataset_path}")
        return results

    # Get all items in directory
    items = sorted(os.listdir(dataset_path))
    
    for folder_name in items:
        full_path = join(dataset_path, folder_name)
        if not isdir(full_path):
            continue
            
        # Check if it looks like a trainer folder (has plans.json)
        if not os.path.exists(join(full_path, 'plans.json')):
            continue
            
        try:
            # Parse folder name: Trainer__Plans__Config
            # nnUNet pads with double underscore
            parts = folder_name.split('__')
            if len(parts) >= 3:
                trainer_name = parts[0]
                plans_name = parts[1]
                configuration_name = parts[2]
            else:
                continue

            plans = load_json(join(full_path, 'plans.json'))
            dataset_json = load_json(join(full_path, 'dataset.json'))
            plans_manager = PlansManager(plans)
            
            try:
                configuration_manager = plans_manager.get_configuration(configuration_name)
            except KeyError:
                print(f"Configuration {configuration_name} not found in plans for {folder_name}")
                continue

            num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
            
            # Find trainer class
            # Search in U-Mamba/umamba/nnunetv2/training/nnUNetTrainer
            search_path = join(nnunetv2.__path__[0], "training", "nnUNetTrainer")
            trainer_class = recursive_find_python_class(search_path, trainer_name, 'nnunetv2.training.nnUNetTrainer')
            
            if trainer_class is None:
                print(f"Could not find trainer class {trainer_name}")
                continue
                
            network = build_network(trainer_class, plans_manager, dataset_json, configuration_manager, num_input_channels)

            if network:
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                network.to(device)
                network.eval()
                
                # Determine input size
                patch_size = configuration_manager.patch_size
                # shape: (1, channels, *patch_size)
                input_size = (1, num_input_channels, *patch_size)
                
                macs, params = calculate_complexity(network, input_size)
                
                # Convert to Millions and Giga
                params_m = params / 1e6
                flops_g = macs / 1e9
                
                print(f"{trainer_name:<60} | {configuration_name:<10} | {params_m:<10.2f} | {flops_g:<10.2f}")
                
                results.append({
                    'Dataset': os.path.basename(dataset_path),
                    'Trainer': trainer_name,
                    'Plans': plans_name,
                    'Configuration': configuration_name,
                    'Params (M)': round(params_m, 4),
                    'FLOPs (G)': round(flops_g, 4)
                })
                
                # Clear memory
                del network
                torch.cuda.empty_cache()
            else:
                 print(f"{trainer_name:<60} | {configuration_name:<10} | {'FAILED':<10} | {'FAILED':<10}")

        except Exception as e:
            # print(f"Error processing {folder_name}: {e}")
            pass 

    return results

if __name__ == "__main__":
    datasets = [
        "/home/chen/seg6/U-Mamba/data/nnUNet_results/Dataset119_Perovskite",
        "/home/chen/seg6/U-Mamba/data/nnUNet_results/Dataset110_Perovskite"
    ]
    
    all_experiments = []
    
    for ds in datasets:
        ds_results = process_dataset(ds)
        all_experiments.extend(ds_results)
        
    output_csv = "model_complexity_stats.csv"
    if all_experiments:
        keys = all_experiments[0].keys()
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_experiments)
        print(f"\nResults saved to {output_csv}")
    else:
        print("No results found.")
