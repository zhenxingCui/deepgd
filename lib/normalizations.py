from .imports import *
from .functions import *
    
    
class ZeroCenter(nn.Module):
    def __init__(self, return_center=False):
        super().__init__()
        self.return_center = return_center
        
    def forward(self, pos, data):
        batch = make_batch(data)
        center = torch_scatter.scatter(pos, batch.batch, dim=0, reduce='mean')
        centered_pos = pos - center[batch.batch]
        if self.return_center:
            return centered_pos, center
        return centered_pos
    

class RescaleByStress(nn.Module):
    def __init__(self, return_scale=False):
        super().__init__()
        self.return_scale = return_scale
        
    def forward(self, pos, data):
        batch = make_batch(data)
        d = batch.full_edge_attr[:, 0]
        start, end = get_full_edges(pos, batch)
        u = (end - start).norm(dim=1)
        index = batch.batch[batch.edge_index[0]]
        scale = torch_scatter.scatter((u/d)**2, index) / torch_scatter.scatter(u/d, index)
        scaled_pos = pos / scale[batch.batch][:, None]
        if self.return_scale:
            return scaled_pos, scale
        return scaled_pos
    

class RescaleByDensity(nn.Module):
    def __init__(self, return_scale=False):
        super().__init__()
        self.return_scale = return_scale
        self.center = ZeroCenter(return_center=True)
        
    def forward(self, pos, data):
        batch = make_batch(data)
        centered_pos, center = self.center(pos, batch)
        radius = torch.linalg.norm(centered_pos, dim=1)
        scale =  torch_scatter.scatter(radius, batch.batch, reduce='mean') / torch.sqrt(batch.n)
        scaled_pos = centered_pos / scale[batch.batch][:, None] + center[batch.batch]
        if self.return_scale:
            return scaled_pos, scale
        return scaled_pos
        
        
class RotateByPCA(nn.Module):
    def __init__(self, angle=0, return_rotation=False):
        super().__init__()
        self.base_rotation = torch.tensor([[-np.sin(angle), np.cos(angle)],
                                           [ np.cos(angle), np.sin(angle)]]).float()
        self.return_rotation = return_rotation
        self.center = ZeroCenter(return_center=True)
        
    def forward(self, pos, data):
        batch = make_batch(data)
        centered_pos, center = self.center(pos, batch)
        XXT = torch.einsum('ni,nj->nij', centered_pos, centered_pos)
        cov = torch_scatter.scatter(XXT, batch.batch, dim=0, reduce='mean')
        rotation = torch.linalg.eigh(cov).eigenvectors
        rotated_pos = torch.einsum('ij,njk,nk->ni', 
                                   self.base_rotation.to(pos.device),
                                   rotation[batch.batch], 
                                   centered_pos) + center[batch.batch]
        if self.return_rotation:
            return rotated_pos, rotation
        return rotated_pos
    
    
class Normalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = RescaleByDensity()
        self.rotate = RotateByPCA()
        self.center = ZeroCenter()
        
    def forward(self, pos, data):
        batch = make_batch(data)
        pos = self.scale(pos, batch)
        pos = self.rotate(pos, batch)
        pos = self.center(pos, batch)
        return pos
    
    
class StressMajorizationAndCenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = RescaleByStress()
        self.center = ZeroCenter()
        
    def forward(self, pos, data):
        batch = make_batch(data)
        pos = self.scale(pos, batch)
        pos = self.center(pos, batch)
        return pos