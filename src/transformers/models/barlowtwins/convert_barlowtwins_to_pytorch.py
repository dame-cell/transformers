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
import logging
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch import Tensor

from transformers import AutoImageProcessor, BarlowTwinsConfig, BarlowTwinsForImageClassification
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger()


@dataclass
class Tracker:
    module: nn.Module
    traced: List[nn.Module] = field(default_factory=list)
    handles: list = field(default_factory=list)

    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor):
        has_not_submodules = len(list(m.modules())) == 1 or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)
        if has_not_submodules:
            self.traced.append(m)

    def __call__(self, x: Tensor):
        for m in self.module.modules():
            self.handles.append(m.register_forward_hook(self._forward_hook))
        self.module(x)
        [x.remove() for x in self.handles]
        return self

    @property
    def parametrized(self):
        # check the len of the state_dict keys to see if we have learnable params
        return list(filter(lambda x: len(list(x.state_dict().keys())) > 0, self.traced))

@dataclass
class ModuleTransfer:
    src: nn.Module
    dest: nn.Module
    verbose: int = 0
    src_skip: List = field(default_factory=list)
    dest_skip: List = field(default_factory=list)

    def __call__(self, x: Tensor):
        """
        Transfer the weights of `self.src` to `self.dest` by performing a forward pass using `x` as input. Under the
        hood we tracked all the operations in both modules.
        """
        dest_traced = Tracker(self.dest)(x).parametrized
        src_traced = Tracker(self.src)(x).parametrized

        src_traced = list(filter(lambda x: type(x) not in self.src_skip, src_traced))
        dest_traced = list(filter(lambda x: type(x) not in self.dest_skip, dest_traced))

        if len(dest_traced) != len(src_traced):
            raise Exception(
                f"Numbers of operations are different. Source module has {len(src_traced)} operations while"
                f" destination module has {len(dest_traced)}."
            )

        for dest_m, src_m in zip(dest_traced, src_traced):
            dest_m.load_state_dict(src_m.state_dict())
            if self.verbose == 1:
                print(f"Transfered from={src_m} to={dest_m}")








def convert_weight_and_push(name: str, config: BarlowTwinsConfig, save_directory: Path, push_to_hub: bool = True):
    print(f"Converting {name}...")
    with torch.no_grad():
        from_model = torch.hub.load('facebookresearch/barlowtwins:main', name)
        our_model = BarlowTwinsForImageClassification(config=config)
        module_transfer = ModuleTransfer(src=from_model, dest=our_model)

        x = torch.randn((1, 3, 224, 224))
        module_transfer(x)

        from_model_outs = from_model(x)
        our_model_outs = our_model(x).logits

        assert torch.allclose(from_model(x), our_model(x).logits), "The model logits don't match the original one."

    checkpoint_name = f"Barlowtwins{'-'.join(name.split('resnet'))}"
    print(checkpoint_name)

    if push_to_hub:
        our_model.push_to_hub(
            repo_id=f"damerajee/{checkpoint_name}",
            commit_message="Add model",
            use_temp_dir=True,
        )

        # we can use the convnext one
        image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
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
            " currently: resnet50. If `None`, all of them will the converted."
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

