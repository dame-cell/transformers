<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# BarlowTwins

## Overview

The BarlowTwins model was proposed in [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) by Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stephane Deny.

<INSERT SHORT SUMMARY HERE>
The Barlow Twins method is an approach to self-supervised learning that aims to learn useful representations of input data without relying on human annotations. It is based on the redundancy-reduction principle, which was first proposed in neuroscience by H. Barlow. The method uses an objective function that measures the cross-correlation matrix between the embeddings of two identical networks fed with distorted versions of a batch of samples, and tries to make this matrix close to the identity matrix. This causes the embedding vectors of distorted versions of a sample to be similar, while minimizing the redundancy between the components of these vectors. The method is conceptually simple, easy to implement, and learns useful representations as opposed to trivial solutions. 


#### The abstract from the paper is the following:

*The abstract from the paper is the following:
Self-supervised learning (SSL) is rapidly closing
the gap with supervised methods on large computer vision benchmarks. A successful approach
to SSL is to learn embeddings which are invariant
to distortions of the input sample. However, a
recurring issue with this approach is the existence
of trivial constant solutions. Most current methods avoid such solutions by careful implementation details. We propose an objective function
that naturally avoids collapse by measuring the
cross-correlation matrix between the outputs of
two identical networks fed with distorted versions
of a sample, and making it as close to the identity
matrix as possible. This causes the embedding
vectors of distorted versions of a sample to be similar, while minimizing the redundancy between
the components of these vectors. The method is
called BARLOW TWINS, owing to neuroscientist
H. Barlow’s redundancy-reduction principle applied to a pair of identical networks. BARLOW
TWINS does not require large batches nor asymmetry between the network twins such as a predictor network, gradient stopping, or a moving
average on the weight updates. Intriguingly it benefits from very high-dimensional output vectors.
BARLOW TWINS outperforms previous methods
on ImageNet for semi-supervised classification in
the low-data regime, and is on par with current
state of the art for ImageNet classification with
a linear classifier head, and for transfer tasks of
classification and object detection*

## Usage tips

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/barlowtwins')
model = AutoModel.from_pretrained('facebook/barlowtwins')

inputs = processor(images=image, return_tensors="pt")
logits = model(**inputs)

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

This model was contributed by [damerajee](https://huggingface.co/damerajee).
The original code can be found [here](https://github.com/facebookresearch/barlowtwins).


## BarlowTwinsConfig

[[autodoc]] BarlowTwinsConfig

<frameworkcontent>
<pt>

## BarlowTwinsModel

[[autodoc]] BarlowTwinsModel
    - forward

## BarlowTwinsForImageClassification

[[autodoc]] BarlowTwinsForImageClassification
    - forward

</pt>
<tf>
