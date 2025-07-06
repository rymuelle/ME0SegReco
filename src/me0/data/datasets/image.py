from pathlib import Path
from typing import cast, Any, Union
import os
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from tensordict import TensorDict
import h5py
import tqdm
from .base import pad_muon, get_total_entries
from .base import TensorDictListDataset

class ME0DigiImageDataset(TensorDictListDataset):
    def __init__(
        self,
        file: str | Path | list[str] | list[Path],
        features: list[dict] | None = None,
        nfiles: int | None = None,
        pad: bool = False,
        layer_first: bool = False,
        get_central_bx: bool = True,
        **kwargs: Any,
    ):
        if layer_first:
            self.image_size = (6, 8, 192) if pad else (6, 8, 384)
            self.keys = ['layer', 'ieta', 'strip']
        else:
            self.image_size = (8, 6, 192) if pad else (8, 6, 384)
            self.keys = ['ieta', 'layer', 'strip']

        self.file = str(file)
        self.nfiles = nfiles
        self.layer_first = layer_first
        self.features = features
        self.get_central_bx = get_central_bx

        data = self.process()
        super().__init__(data)

    def process(self) -> list[TensorDict]:
        tree_iter = h5py.File(self.file, 'r') 

#        total = get_total_entries(tree_iter, self.nfiles)

        data: list[TensorDict] = []
        i = 0
        for _, chunk in (pbar := tqdm.tqdm(tree_iter.items(), total=self.nfiles)):
            if isinstance(chunk, h5py.Dataset):
                chunk = cast(dict[str, np.ndarray], chunk)
                chunk_size = len(chunk['pt'])
                pbar.set_description(f'processing {chunk_size} events')
                if self.get_central_bx:
                    for example in chunk:
                        data.append(self.process_example(example))
                else:
                    data += self.process_chunk(chunk)
#                pbar.update(n=chunk_size)
                i += 1
                pbar.update(n=i)
                if (i == self.nfiles): break
        return data

    def process_example(self, example) -> TensorDict:
        layer = torch.tensor(example['layer'])
        ieta = torch.tensor(example['ieta'])
        strip = torch.tensor(example['strip'])
        bx = torch.tensor(example['bx'])
        label = torch.tensor(example['label'])
        
        same_layer = (layer[:, None] == layer[None, :])
        same_ieta = (ieta[:, None] == ieta[None, :])
        same_strip = (strip[:, None] == strip[None, :])
        
        same_pos = (same_layer & same_ieta & same_strip)
        same_pos.fill_diagonal_(False)
        same_pos = same_pos.nonzero()

        processed = []
        select_mask = torch.ones_like(label, dtype=torch.bool)
        for indices in same_pos:
            if indices.tolist() in processed: continue
            if label[indices].any():
                idx = label[indices].argmin()
            else:
                idx = abs(bx[indices]).argmax()
            select_mask[indices[idx]] = False
            processed.append(indices.flipud().tolist())

        layer = layer[select_mask]
        ieta = ieta[select_mask]
        strip = strip[select_mask]
        label = label[select_mask]

        if self.layer_first:
            indices = torch.stack([layer, ieta, strip], axis=0)
        else:
            indices = torch.stack([ieta, layer, strip], axis=0)

        output = {'indices': indices, 'target': label}
        output |= {key: example[key] for key in ['pt', 'eta', 'phi']}

        if self.features:
            for key, v in self.features.items():
                output |= {
                    key: torch.tensor((example[key] - v['min'])/(v['max'] - v['min']), dtype=torch.float32)[select_mask]
                }
                
        output = TensorDict(
            source={key: output[key] for key in output.keys()},
            batch_size=[]
        )

        return output

    def process_chunk(
        self,
        input: dict[str, npt.NDArray],
    ) -> list[TensorDict]:
        indices_chunk = zip(*[input[key] for key in self.keys])
        indices_chunk = [torch.from_numpy(np.stack(each, axis=0)) for each in indices_chunk]
        target_chunk = [torch.from_numpy(each).type(torch.int32) for each in input['label']]

        output = {'indices': indices_chunk, 'target': target_chunk}
        output |= {key: pad_muon(input[key]) for key in ['pt', 'eta', 'phi']}

        if self.features:
            output |= {key: [torch.from_numpy((each - v['min'])/(v['max'] - v['min'])).type(torch.float32) 
                             for each in input[key]] for key, v in self.features.items()}

        chunk_size = len(output['target'])
        output = [
            TensorDict(
                source={key: output[key][idx] for key in output.keys()},
                batch_size=[]
            )
            for idx in range(chunk_size)
        ]
        return output

    def to_image(self, indices: Tensor, values: Tensor) -> Tensor:
        image = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=self.image_size
        )
        image = image.to_dense()
        return image

    def __getitem__(self, index: int | slice) -> Union[TensorDict, 'ME0DigiImageDataset']:
        if isinstance(index, slice):
            return self.__class__.from_data(
                    self.data[index],
                    features=self.features,
                    layer_first=self.layer_first,
                    get_central_bx=self.get_central_bx,
                    image_size=self.image_size,
                    keys=self.keys,
                    )
                    
        example = self.data[index].clone()

        indices = example.pop('indices')
        target = example.pop('target')
        input = torch.ones_like(target)

        target = self.to_image(indices=indices, values=target)
        input = self.to_image(indices=indices, values=input)

        example['target'] = target
        example['input'] = input
        example['data_mask'] = input.gt(0)

        if self.features:
            example['input'] = torch.stack([example['input']] + [self.to_image(indices=indices, values=example.pop(key)) for key in self.features.keys()])
        return example

    @classmethod
    def from_data(cls, data: list[TensorDict], features=None, **kwargs) -> 'ME0DigiImageDataset':
        obj = cls.__new__(cls)
        super(cls, obj).__init__(data)
        obj.file = None
        obj.nfiles = None
        obj.features = features
        obj.layer_first = kwargs.get('layer_first', False)
        obj.get_central_bx = kwargs.get('get_central_bx', True)
        obj.image_size = kwargs.get('image_size')
        obj.keys = kwargs.get('keys')
        return obj
