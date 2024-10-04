# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Barlowtwins checkpoints from torch hub."""

import argparse
import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List

import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch import Tensor

from transformers import AutoImageProcessor, BarlowTwinsForImageClassification , BarlowTwinsConfig , BarlowTwinsModel
from transformers.utils import logging
from PIL import Image 
import requests 


logging.set_verbosity_info()
logger = logging.get_logger()


@dataclass
class Tracker:
    module: nn.Module
    traced: List[nn.Module] = field(default_factory=list)
    handles: list = field(default_factory=list)
    module_names: List[str] = field(default_factory=list)  # Added to track module names

    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor):
        has_not_submodules = len(list(m.modules())) == 1 or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)
        if has_not_submodules:
            self.traced.append(m)
            # Store the full path of the module
            for name, module in self.module.named_modules():
                if module is m:
                    self.module_names.append(name)
                    break

    def __call__(self, x: Tensor):
        for m in self.module.modules():
            self.handles.append(m.register_forward_hook(self._forward_hook))
        self.module(x)
        [x.remove() for x in self.handles]
        return self

    @property
    def parametrized(self):
        return list(filter(lambda x: len(list(x.state_dict().keys())) > 0, self.traced))

    def print_module_info(self):
        """Print detailed information about traced modules"""
        print(f"\nModule Structure for {type(self.module).__name__}:")
        for i, (module, name) in enumerate(zip(self.traced, self.module_names)):
            print(f"\n{i+1}. Module: {name}")
            print(f"   Type: {type(module).__name__}")
            print(f"   Parameters: {sum(p.numel() for p in module.parameters())}")
            for param_name, param in module.named_parameters():
                print(f"   - {param_name}: {tuple(param.shape)}")

@dataclass
class ModuleTransfer:
    src: nn.Module
    dest: nn.Module
    verbose: int = 0
    src_skip: List = field(default_factory=list)
    dest_skip: List = field(default_factory=list)

    def verify_weights(self, src_m: nn.Module, dest_m: nn.Module) -> bool:
        """Verify if weights were transferred correctly"""
        src_state = src_m.state_dict()
        dest_state = dest_m.state_dict()
        
        if src_state.keys() != dest_state.keys():
            print(f"\nWarning: Different state dict keys!")
            print(f"Source keys: {list(src_state.keys())}")
            print(f"Destination keys: {list(dest_state.keys())}")
            return False
        
        all_match = True
        for key in src_state.keys():
            if not torch.allclose(src_state[key], dest_state[key], rtol=1e-4, atol=1e-4):
                print(f"\nMismatch in {key}:")
                print(f"Max difference: {(src_state[key] - dest_state[key]).abs().max().item()}")
                print(f"Mean difference: {(src_state[key] - dest_state[key]).abs().mean().item()}")
                print(f"Source shape: {src_state[key].shape}")
                print(f"Dest shape: {dest_state[key].shape}")
                all_match = False
        return all_match

    def __call__(self, x: Tensor):
        print("\nStarting weight transfer verification...")
        
        # Track modules
        dest_tracker = Tracker(self.dest)
        src_tracker = Tracker(self.src)
        
        # Get traced modules
        dest_traced = dest_tracker(x).parametrized
        src_traced = src_tracker(x).parametrized

        # Print detailed module information
        print("\nSource model structure:")
        src_tracker.print_module_info()
        print("\nDestination model structure:")
        dest_tracker.print_module_info()

        # Filter modules
        src_traced = list(filter(lambda x: type(x) not in self.src_skip, src_traced))
        dest_traced = list(filter(lambda x: type(x) not in self.dest_skip, dest_traced))

        if len(dest_traced) != len(src_traced):
            raise Exception(
                f"Numbers of operations are different.\n"
                f"Source module has {len(src_traced)} operations: {[type(m).__name__ for m in src_traced]}\n"
                f"Destination module has {len(dest_traced)} operations: {[type(m).__name__ for m in dest_traced]}"
            )

        all_transfers_verified = True
        for i, (dest_m, src_m) in enumerate(zip(dest_traced, src_traced)):
            print(f"\nTransferring weights for module pair {i+1}:")
            print(f"Source: {type(src_m).__name__}")
            print(f"Destination: {type(dest_m).__name__}")
            
            if type(src_m) != type(dest_m):
                print(f"Warning: Module types don't match!")
            
            # Transfer weights
            dest_m.load_state_dict(src_m.state_dict())
            
            # Verify transfer
            if not self.verify_weights(src_m, dest_m):
                print(f"Warning: Weight verification failed for module pair {i+1}")
                all_transfers_verified = False
            else:
                print(f"✓ Weight transfer verified for module pair {i+1}")

            if self.verbose == 1:
                print(f"Transferred from={src_m} to={dest_m}")

        if all_transfers_verified:
            print("\n✓ All weight transfers verified successfully!")
        else:
            print("\n⚠ Some weight transfers could not be verified!")
            print("This might be due to architectural differences or normalization layers.")


import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def transform_image(image: Image.Image) -> torch.Tensor:
    # Parameters from ConvNextImageProcessor
    crop_pct = 0.875
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    size = 224

    crop_size = int(size / crop_pct)

    transform = transforms.Compose([
        transforms.Resize(crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ])
    
    transformed_image = transform(image)
    return transformed_image

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convert_weight_and_push(name: str, config: BarlowTwinsConfig, save_directory: Path, push_to_hub: bool = True):
    print(f"Converting {name}...")
    with torch.no_grad():
        from_model = torch.hub.load('facebookresearch/barlowtwins:main', name)
        from_model = nn.Sequential(*(list(from_model.children())[:-1])) 
 
        our_model = BarlowTwinsModel(config=config)

        print("barlowtwins architecture:",our_model)
        print("resnet50 architecture:",from_model)
        
        print("from_model paramtere count",count_parameters(from_model))
        print("our_model paramtere count",count_parameters(our_model))


        module_transfer = ModuleTransfer(src=from_model, dest=our_model,verbose=1)
        x = torch.randn((1, 3, 224, 224))
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        from datasets import load_dataset

        dataset = load_dataset("huggingface/cats-image")
        image = dataset["test"]["image"][0]
        inputs = processor(image, return_tensors="pt").pixel_values
        from_inputs = transform_image(image).unsqueeze(0)
        # Convert inputs to PyTorch tensor if it's not already
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs)
        
        # Ensure both tensors are on the same device
        from_inputs = from_inputs.to(inputs.device)
        
        print("from_inputs shape:", from_inputs.shape)
        print("inputs shape:", inputs.shape)
        
        # Print some sample values
        print("Sample values from from_inputs:")
        print(from_inputs[0, :3, :3, :3])
        print("Sample values from inputs:")
        print(inputs[0, :3, :3, :3])
        
        assert torch.allclose(from_inputs, inputs[0]), "The pixel values do not match somehow"
        


        from_model_out = from_model(from_inputs)
        print("from_model_out",from_model_out.size())
        our_model_out = our_model(inputs)
        print("our_model_out",our_model_out.pooler_output.size())




        print("our_model_out ", our_model_out.pooler_output)
        print("from_model_out", from_model_out)
     
    # Use a higher tolerance due to potential differences in implementation
    assert torch.allclose(from_model_out, our_model_out.pooler_output), "The model logits don't match the original one."

    checkpoint_name = f"resnet{'-'.join(name.split('resnet'))}"
    print(checkpoint_name)

    if push_to_hub:
        our_model.push_to_hub(
            repo_id=f"damerajee/{checkpoint_name}",
            commit_message="Add model",
            use_temp_dir=True,
        )

        # we can use the convnext one
        image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k-1k")
        image_processor.push_to_hub(
            repo_id=f"damerajee/{checkpoint_name}",
            commit_message="Add image processor",
            use_temp_dir=True,
        )

        print(f"Pushed {checkpoint_name}")



def convert_weights_and_push(save_directory: Path, model_name: str = None, push_to_hub: bool = True):
    filename = "imagenet-1k-id2label.json"
    num_labels = 1000

    # Load ImageNet labels
    repo_id = "huggingface/label-files"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    
    ImageNetPreTrainedConfig = partial(BarlowTwinsConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)

    names_to_config = {

        "resnet50": ImageNetPreTrainedConfig(
            depths=[3, 4, 6, 3], hidden_sizes=[256, 512, 1024, 2048],embedding_size=64, layer_type="bottleneck"
        ),
    }

    convert_weight_and_push(model_name, names_to_config[model_name], save_directory, push_to_hub)    
    return names_to_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help=(
            "The name of the model you wish to convert, it must be one of the supported resnet* architecture,"
            " currently: resnet18,26,34,50,101,152. If `None`, all of them will the converted."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=Path,
        required=True,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        default=False,
        type=bool,
        required=False,
        help="If True, push model and image processor to the hub.",
    )

    args = parser.parse_args()
    pytorch_dump_folder_path: Path = args.pytorch_dump_folder_path
    pytorch_dump_folder_path.mkdir(exist_ok=True, parents=True)
    convert_weights_and_push(pytorch_dump_folder_path,args.model_name,args.push_to_hub)

