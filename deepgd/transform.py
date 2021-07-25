from ._dependencies import *

from .functions import *
    
      
class IdentityTransformation(nn.Module):
    def __init__(self, *args, **kvargs):
        super().__init__()
        
    def forward(self, pos, data):
        return pos
    

class LinearTransformation(nn.Module):
    def __init__(self, transformation_matrix=torch.eye(2)):
        super().__init__()
        self.transformation = transformation_matrix.float()
        
    def forward(self, pos, data):
        pos = torch.einsum('ij,nj->ni', 
                           self.transformation.to(pos.device),
                           pos)
        return pos
    
    
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
    def __init__(self, scale_factor=1, return_scale=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.return_scale = return_scale
        
    def forward(self, pos, data):
        batch = make_batch(data)
        d = batch.full_edge_attr[:, 0]
        start, end = get_full_edges(pos, batch)
        u = (end - start).norm(dim=1)
        index = batch.batch[batch.edge_index[0]]
        scale = torch_scatter.scatter((u/d)**2, index) / torch_scatter.scatter(u/d, index)
        scaled_pos = self.scale_factor * pos / scale[batch.batch][:, None]
        if self.return_scale:
            return scaled_pos, scale
        return scaled_pos
    

class RescaleByDensity(nn.Module):
    def __init__(self, scale_factor=1, return_scale=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.return_scale = return_scale
        
    def forward(self, pos, data):
        batch = make_batch(data)
        radius = torch.linalg.norm(pos, dim=1)
        scale =  torch_scatter.scatter(radius, batch.batch, reduce='mean') / torch.sqrt(batch.n)
        scaled_pos = self.scale_factor * pos / scale[batch.batch][:, None]
        if self.return_scale:
            return scaled_pos, scale
        return scaled_pos
    
    
class RescaleByMinMax(nn.Module):
    def __init__(self, scale_factor=1, return_scale=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.return_scale = return_scale
        
    def forward(self, pos, data):
        batch = make_batch(data)
        x, y = pos[:, 0], pos[:, 1]
        xmin = torch_scatter.scatter(x, batch.batch, reduce='min')
        xmax = torch_scatter.scatter(x, batch.batch, reduce='max')
        ymin = torch_scatter.scatter(y, batch.batch, reduce='min')
        ymax = torch_scatter.scatter(y, batch.batch, reduce='max')
        xrange = xmax - xmin
        yrange = ymax - ymin
        scale = torch.maximum(xrange, yrange)
        scaled_pos = self.scale_factor * pos / scale[batch.batch][:, None]
        if self.return_scale:
            return scaled_pos, scale
        return scaled_pos
        
        
class RotateByPCA(nn.Module):
    def __init__(self, base_angle=0, return_rotation=False):
        super().__init__()
        sin = np.sin(base_angle)
        cos = np.cos(base_angle)
        self.base_rotation = torch.tensor([[-sin, cos],
                                           [ cos, sin]]).float()
        self.return_rotation = return_rotation
        
    def forward(self, pos, data):
        batch = make_batch(data)
        XXT = torch.einsum('ni,nj->nij', pos, pos)
        cov = torch_scatter.scatter(XXT, batch.batch, dim=0, reduce='mean')
        rotation = torch.linalg.eigh(cov).eigenvectors
        rotated_pos = torch.einsum('ij,njk,nk->ni', 
                                   self.base_rotation.to(pos.device),
                                   rotation[batch.batch], 
                                   pos)
        if self.return_rotation:
            return rotated_pos, rotation
        return rotated_pos


class CanonicalNormalization(nn.Module):
    def __init__(self, center=ZeroCenter, rotate=RotateByPCA, scale=RescaleByDensity, base_angle=0, scale_factor=1):
        super().__init__()
        self.center = center()
        self.rotate = rotate(base_angle=base_angle)
        self.scale = scale(scale_factor=scale_factor)
        
    def forward(self, pos, data):
        batch = make_batch(data)
        pos = self.center(pos, batch)
        pos = self.rotate(pos, batch)
        pos = self.scale(pos, batch)
        return pos
    
    
class CenteredStressMajorization(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize = CanonicalNormalization(rotate=IdentityTransformation,
                                                scale=RescaleByStress)
        
    def forward(self, pos, data):
        batch = make_batch(data)
        pos = self.normalize(pos, batch)
        return pos
    
