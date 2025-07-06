from pathlib import Path
from typing import cast, Any
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
import tensordict as td
from tensordict import TensorDict
import h5py
import tqdm
from .base import get_total_entries, TensorDictListDataset


class ME0IndexDataset(TensorDictListDataset):

    def __init__(
        self,
        file: str | Path | list[str] | list[Path],
        features: list[dict] | None = None,
        nfiles: int | None = None,
        pad: bool = False,
        layer_first: bool = False,
        **kwargs: Any,
    ):
        if layer_first:
            self.scale_factor = [5, 7, 192] if pad else [5, 7, 383]
            self.keys = ['layer', 'ieta', 'strip']
        else:
            self.scale_factor = [7, 5, 192] if pad else [7, 5, 383]
            self.keys = ['ieta', 'layer', 'strip']

        self.scale_factor = torch.tensor(self.scale_factor)
        self.file = str(file)
        self.nfiles = nfiles
        self.features = features

        data = self.process()
        super().__init__(data)

    def process(self) -> list[TensorDict]:
        tree_iter = h5py.File(self.file, 'r') 
        total = get_total_entries(tree_iter, self.nfiles)

        data: list[TensorDict] = []
        i = 0
        for _, chunk in (pbar := tqdm.tqdm(tree_iter.items(), total=total)):
            if isinstance(chunk, h5py.Dataset):
                chunk = cast(dict[str, np.ndarray], chunk)
                chunk_size = len(chunk['pt'])
                pbar.set_description(f'processing {chunk_size} events')
                data += self.process_chunk(chunk)
                pbar.update(n=chunk_size)
                i += 1
                if (i == self.nfiles): break
        return data

    def process_chunk(
        self,
        input: dict[str, npt.NDArray],
    ) -> list[TensorDict]:
        indices_chunk = zip(*[input[key] for key in self.keys])
        indices_chunk = [torch.from_numpy(np.stack(each, axis=1)) for each in indices_chunk]
        target_chunk = [torch.from_numpy(each).type(torch.int32) for each in input['label']]
        input_chunk = [each/self.scale_factor for each in indices_chunk]

        if self.features:
            features_chunk = zip(*[((input[key] - v['min']) / (v['max'] - v['min'])) for key, v in self.features.items()])
            features_chunk = [torch.from_numpy(np.stack(each, axis=1)) for each in features_chunk]
            input_chunk = [torch.hstack(each).type(torch.float32) for each in zip(input_chunk, features_chunk)]


        output = {
            'indices': indices_chunk,
            'input': input_chunk,
            'target': target_chunk,
        }

        for key in ['pt', 'eta', 'phi']:
            output |= {key: [torch.from_numpy(each).type(torch.float32) for each in input[key]]}

        chunk_size = len(output['indices'])
        output = [
            TensorDict(
                source={key: output[key][idx] for key in output.keys()},
                batch_size=[]
            )
            for idx in range(chunk_size)
        ]

        return output

    def collate(self, batch: list[TensorDict]) -> TensorDict:
        batch = td.pad_sequence(
            list_of_tensordicts=batch,
            pad_dim=0,
            padding_value=0,
            return_mask=True)
        masks = batch.pop('masks')
        batch.set('data_mask', masks.pop('input'))
        batch.set('muon_mask', masks.pop('pt'))
        return batch 
