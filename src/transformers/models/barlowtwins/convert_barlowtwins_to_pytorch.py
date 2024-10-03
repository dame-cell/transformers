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

from transformers import AutoImageProcessor, BarlowTwinsConfig, BarlowTwinsForImageClassification
from transformers.utils import logging
from PIL import Image 
import requests 


logging.set_verbosity_info()
logger = logging.get_logger()
import warnings
warnings.filterwarnings("ignore")

def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image

def create_resnet_to_barlowtwins_rename_keys():
    rename_keys = []
    
    # Initial layers
    rename_keys.append(("conv1.weight", "barlowtwins.embedder.embedder.convolution.weight"))
    rename_keys.append(("bn1.weight", "barlowtwins.embedder.embedder.normalization.weight"))
    rename_keys.append(("bn1.bias", "barlowtwins.embedder.embedder.normalization.bias"))
    rename_keys.append(("bn1.running_mean", "barlowtwins.embedder.embedder.normalization.running_mean"))
    rename_keys.append(("bn1.running_var", "barlowtwins.embedder.embedder.normalization.running_var"))
    rename_keys.append(("bn1.num_batches_tracked", "barlowtwins.embedder.embedder.normalization.num_batches_tracked"))

    # Encoder stages
    for stage in range(4):
        num_layers = 3 if stage == 0 else 4 if stage == 1 else 6 if stage == 2 else 3
        for layer in range(num_layers):
            base_resnet = f"layer{stage+1}.{layer}"
            base_barlowtwins = f"barlowtwins.encoder.stages.{stage}.layers.{layer}"

            # Shortcut (downsample) layers
            if layer == 0:
                rename_keys.append((f"{base_resnet}.downsample.0.weight", f"{base_barlowtwins}.shortcut.convolution.weight"))
                rename_keys.append((f"{base_resnet}.downsample.1.weight", f"{base_barlowtwins}.shortcut.normalization.weight"))
                rename_keys.append((f"{base_resnet}.downsample.1.bias", f"{base_barlowtwins}.shortcut.normalization.bias"))
                rename_keys.append((f"{base_resnet}.downsample.1.running_mean", f"{base_barlowtwins}.shortcut.normalization.running_mean"))
                rename_keys.append((f"{base_resnet}.downsample.1.running_var", f"{base_barlowtwins}.shortcut.normalization.running_var"))
                rename_keys.append((f"{base_resnet}.downsample.1.num_batches_tracked", f"{base_barlowtwins}.shortcut.normalization.num_batches_tracked"))

            # Convolutions and Batch Norms
            for i in range(3):
                rename_keys.append((f"{base_resnet}.conv{i+1}.weight", f"{base_barlowtwins}.layer.{i}.convolution.weight"))
                rename_keys.append((f"{base_resnet}.bn{i+1}.weight", f"{base_barlowtwins}.layer.{i}.normalization.weight"))
                rename_keys.append((f"{base_resnet}.bn{i+1}.bias", f"{base_barlowtwins}.layer.{i}.normalization.bias"))
                rename_keys.append((f"{base_resnet}.bn{i+1}.running_mean", f"{base_barlowtwins}.layer.{i}.normalization.running_mean"))
                rename_keys.append((f"{base_resnet}.bn{i+1}.running_var", f"{base_barlowtwins}.layer.{i}.normalization.running_var"))
                rename_keys.append((f"{base_resnet}.bn{i+1}.num_batches_tracked", f"{base_barlowtwins}.layer.{i}.normalization.num_batches_tracked"))

    # Final fully connected layer (if applicable)
    rename_keys.append(("fc.weight", "classifier.1.weight"))
    rename_keys.append(("fc.bias", "classifier.1.bias"))

    return rename_keys



def transfer_weights(resnet50_state_dict, barlowtwins_state_dict):
    rename_keys = create_resnet_to_barlowtwins_rename_keys()
    new_state_dict = {}

    def __call__(self, x: Tensor):
        for m in self.module.modules():
            self.handles.append(m.register_forward_hook(self._forward_hook))
        self.module(x)
        [x.remove() for x in self.handles]
        return self

    # Check for keys in BarlowTwins that are not in our mapped state dict
    for key in barlowtwins_state_dict.keys():
        if key not in new_state_dict:
            print(f"Key in BarlowTwins model not mapped from ResNet50: {key}")
            # Initialize these keys with zeros or random values
            new_state_dict[key] = torch.zeros_like(barlowtwins_state_dict[key])

    return new_state_dict


def convert_weight_and_push(name: str, config: BarlowTwinsConfig, save_directory: Path, push_to_hub: bool = True):
    print(f"Converting {name}...")
    with torch.no_grad():
        from_model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        our_model = BarlowTwinsForImageClassification(config).eval()
      
        new_state_dict = transfer_weights(from_model.state_dict(), our_model.state_dict())
        incompatible_keys = our_model.load_state_dict(new_state_dict, strict=False)

        if incompatible_keys.missing_keys:
            print("Missing keys:", incompatible_keys.missing_keys)
        if incompatible_keys.unexpected_keys:
            print("Unexpected keys:", incompatible_keys.unexpected_keys)

        x = torch.randn((1, 3, 224, 224))
        

        resnet_features = from_model(x)
        # Get the output from our BarlowTwins model
        barlowtwins_output = our_model(x)
        

        print("ResNet50 features shape:", resnet_features.shape)
        print("BarlowTwins output shape", barlowtwins_output.logits.shape)
        print("ResNet50 features (first 5):", resnet_features[0, :5])
        print("BarlowTwins output (first 5):", barlowtwins_output.logits[0, :5])
        
        # Calculate the maximum absolute difference
        max_diff = torch.max(torch.abs(resnet_features - barlowtwins_output.logits))
        print("Maximum absolute difference:", max_diff.item())
        
        # Calculate the mean squared error
        mse = torch.mean((resnet_features - barlowtwins_output.logits) ** 2)
        print("Mean Squared Error:", mse.item())
            
    # Use a higher tolerance due to potential differences in implementation
    assert torch.allclose(resnet_features, barlowtwins_output.logits), "The model outputs don't match the original one."

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
    expected_shape = (1, num_labels)

    repo_id = "huggingface/label-files"
    num_labels = num_labels
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    id2label = id2label
    label2id = {v: k for k, v in id2label.items()}

    ImageNetPreTrainedConfig = partial(BarlowTwinsConfig, num_labels=num_labels, id2label=id2label, label2id=label2id)

    names_to_config = {

        "resnet50": ImageNetPreTrainedConfig(
            depths=[3, 4, 6, 3], hidden_sizes=[256, 512, 1024, 2048], layer_type="bottleneck"
        ),
    }

    if model_name:
        convert_weight_and_push(model_name, names_to_config[model_name], save_directory, push_to_hub)
   


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
    convert_weights_and_push(pytorch_dump_folder_path, args.model_name, args.push_to_hub)