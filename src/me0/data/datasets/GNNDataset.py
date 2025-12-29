import h5py
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
import torch



class GNNDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None,
                 d_strip_max=20, d_ieta_max=2, d_bx_max=1):
        self.d_strip_max = d_strip_max
        self.d_ieta_max = d_ieta_max
        self.d_bx_max = d_bx_max

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)


    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []        
        with h5py.File('../data/step4_0.h5', 'r') as f:
            chambers = f['chamber']
            for idx in range(len(chambers)):
                grp = chambers[idx]
                
                # Convert to tensors
                x = torch.stack([
                    torch.tensor(grp['bx'], dtype=torch.float32),
                    torch.tensor(grp['ieta'], dtype=torch.float32),
                    torch.tensor(grp['layer'], dtype=torch.float32),
                    torch.tensor(grp['strip'], dtype=torch.float32),
                    torch.tensor(grp['cls'], dtype=torch.float32)
                ], dim=1)
                
                y = torch.tensor(grp['label']).unsqueeze(1)
                num_nodes = x.size(0)

                # Get indices for all possible pairs (i, j) where i < j
                adj_idx = torch.combinations(torch.arange(num_nodes), r=2)
                i, j = adj_idx[:, 0], adj_idx[:, 1]

                # Calculate differences for all pairs at once
                # This will be used to build the graph and edge attributes
                diffs = torch.abs(x[i] - x[j])

                # Only connect nodes that are close in strip, eta, bx and on different layers
                mask = (diffs[:, 3] <= self.d_strip_max) & \
                       (diffs[:, 1] <= self.d_ieta_max) & \
                       (diffs[:, 0] <= self.d_bx_max) & \
                       (x[i, 2] != x[j, 2]) # Different layers
                
                # The edge attributes will be the differences in the nodes
                # More advanced edge attributes should be considered beyond this demo code
                edge_attr = diffs[mask].t().contiguous().permute(1, 0)  
                edge_index = adj_idx[mask].t().contiguous()
                
                # By adding i,j and j,i, we can make the graph bi-directional
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

                data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
                data_list.append(data)

        # Save the processed data to disk
        torch.save(self.collate(data_list), self.processed_paths[0])



class GNNDatasetTest(Dataset):
    def __init__(self, 
                nbx=7,  bx_min=-3,
                nstrip=384,  
                nieta=8, 
                nlayer=6,
                cls_norm = 6,
                delta_strip = 20, delta_ieta = 2, delta_bx = 1,
                 ):
        '''
        I wrote this dataset as a sanity check. Both result in the same output. 
        '''
        super().__init__()
        self.f = h5py.File('../data/step4_0.h5', 'r')

        self.bx_min = bx_min
        self.nbx = nbx
        self.nstrip = nstrip
        self.nieta = nieta
        self.nlayer = nlayer
        self.cls_norm = cls_norm

        self.delta_strip = delta_strip
        self.delta_ieta = delta_ieta
        self.delta_bx = delta_bx
        
    def __len__(self):
        return len(self.f['chamber'])

    def __getitem__(self, idx):
        grp =  self.f['chamber'][idx]
        strip = torch.tensor(grp['strip'], dtype=torch.float32)
        ieta = torch.tensor(grp['ieta'], dtype=torch.float32)
        layer = torch.tensor(grp['layer'], dtype=torch.float32)
        bx = torch.tensor(grp['bx'], dtype=torch.float32)
        cls = torch.tensor(grp['cls'], dtype=torch.float32)
        label = torch.tensor(grp['label'])

        x = torch.cat(
            [bx.unsqueeze(1), ieta.unsqueeze(1), layer.unsqueeze(1), strip.unsqueeze(1), cls.unsqueeze(1)],
            dim = 1
        )
        y = label.unsqueeze(1)
        

        edge_index = []
        edge_attributes = []
        # Define graph edges
        for i in range(len(strip)):
            for j in range(i+1, len(strip)):
               delta_strip = abs(strip[i]- strip[j])
               if delta_strip > self.delta_strip: continue
               delta_ieta = abs(ieta[i]- ieta[j])
               if delta_ieta > self.delta_ieta: continue
               delta_bx = abs(bx[i]- bx[j])
               if delta_bx > self.delta_bx: continue

               delta_layer = -layer[i] + layer[j]
               if delta_layer==0: continue   
               delta_cls =   abs(cls[i] - cls[j])     
               edge_index.append([i, j])
               edge_index.append([j, i])
               edge_attributes.append([delta_strip , 
                                       delta_ieta , 
                                       delta_bx , 
                                       delta_layer , 
                                       delta_cls])
               edge_attributes.append([delta_strip , 
                                       delta_ieta , 
                                       delta_bx , 
                                       delta_layer , 
                                       delta_cls])

        edge_index = torch.tensor(edge_index).permute(1, 0)
        edge_attributes = torch.tensor(edge_attributes)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attributes, y=y)
        return data