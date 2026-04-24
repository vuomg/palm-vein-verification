import os
import shutil
import glob

src_dir = r"c:\Research\Research\PalmVein"
dest_dir = r"c:\Research\Research\PalmVein\SCAMobileNet"

items_to_copy = [
    # Main script
    "train.py",
    
    # Required Packages
    "biometric",
    
    # Models required by train.py
    "core",                          # RSNet
    "SCA_MobileNet",                 # SCA-MobileNet & MobileNetV3_UIB
    "VeinKAN",                       # VeinKAN
    "FGFNet",                        # FGFNet
    "GSCL_2024",                     # GSCL
    "GSCL-PyTorch",                  # GSCL-PyTorch (missing previously)
    "Modified Densenet-161_2021",    # EUSIPCO 2020 (DenseNet, ResNext, MNASNet)
    "MPSNet_2022",                   # MPSNet
    
    # Utils and Pipeline scripts
    "requirements.txt",
    "dataset_splitter.py",
    "palm_vein_enhancement.py",
    "palm_vein_preprocessing.py",
    "palm_vein_processing.py",
    "prepare_tongji_dataset.py",
    "split_tongji_openset.py",
    "cross_domain_eval.py",
]

# Get relevant bash/batch scripts
bat_files_to_copy = [
    "run_ablation_ca.bat",
    "run_ablation_spp.bat",
    "run_ablation_stn.bat",
    "run_cross_domain_eval.bat",
    "run_gscl.bat",
    "run_tongji_baselines.bat",
    "run_tongji_densenet161.bat",
    "run_tongji_fgfnet.bat",
    "run_tongji_pipeline.bat",
    "run_tongji_train.bat",
    "run_train_sca.bat",
]

items_to_copy.extend(bat_files_to_copy)

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for item in items_to_copy:
    src_item = os.path.join(src_dir, item)
    dest_item = os.path.join(dest_dir, item)
    
    if os.path.exists(src_item):
        if os.path.isdir(src_item):
            print(f"Copying directory: {item}")
            shutil.copytree(src_item, dest_item, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
        else:
            print(f"Copying file: {item}")
            shutil.copy2(src_item, dest_item)
    else:
        print(f"Warning: {item} not found in source directory.")

print("Done creating SCAMobileNet repository folder.")
