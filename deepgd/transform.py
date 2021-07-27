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
    
        
class RotateByPrincipalComponents(nn.Module):
    def __init__(self, base_angle=0, return_components=False):
        super().__init__()
        sin = np.sin(base_angle)
        cos = np.cos(base_angle)
        self.base_rotation = torch.tensor([[-sin, cos],
                                           [ cos, sin]]).float()
        self.return_components = return_components
        
    def forward(self, pos, data):
        batch = make_batch(data)
        XXT = torch.einsum('ni,nj->nij', pos, pos)
        cov = torch_scatter.scatter(XXT, batch.batch, dim=0, reduce='mean')
        components = torch.linalg.eigh(cov).eigenvectors
        rotated_pos = torch.einsum('ij,njk,nk->ni', 
                                   self.base_rotation.to(pos.device),
                                   components[batch.batch], 
                                   pos)
        if self.return_components:
            return rotated_pos, components
        return rotated_pos


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

    
class Standardization(nn.Module):
    def __init__(self, norm_ord=2, scale_factor=1, return_normalizer=False):
        super().__init__()
        self.norm_ord = norm_ord
        self.scale_factor = scale_factor
        self.return_normalizer = return_normalizer
    
    def forward(self, pos, data):
        batch = make_batch(data)
        center = torch_scatter.scatter(pos, batch.batch, dim=0, reduce='mean')
        centered_pos = pos - center[batch.batch]
        square = centered_pos.square()
        var = torch_scatter.scatter(square, batch.batch, dim=0, reduce='mean')
        std = var.sqrt()
        normalizer = torch.linalg.norm(std, ord=self.norm_ord, dim=1)
        scaled_pos = self.scale_factor * pos / normalizer[batch.batch][:, None]
        if self.return_normalizer:
            return scaled_pos, normalizer
        return scaled_pos
    
    
class Normalization(nn.Module):
    def __init__(self, norm_ord=2, scale_factor=1, return_normalizer=False):
        super().__init__()
        self.norm_ord = norm_ord
        self.scale_factor = scale_factor
        self.return_normalizer = return_normalizer
        
    def forward(self, pos, data):
        batch = make_batch(data)
        min = torch_scatter.scatter(pos, batch.batch, dim=0, reduce='min')
        max = torch_scatter.scatter(pos, batch.batch, dim=0, reduce='max')
        range = max - min
        normalizer = torch.linalg.norm(range, ord=self.norm_ord, dim=1)
        scaled_pos = self.scale_factor * pos / normalizer[batch.batch][:, None]
        if self.return_normalizer:
            return scaled_pos, normalizer
        return scaled_pos
           

class ScaleByGraphOrder(nn.Module):
    def __init__(self, scale_factor=1, return_scale=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.return_scale = return_scale
        
    def forward(self, pos, data):
        batch = make_batch(data)
        d = pos.shape[-1]
        scale = batch.n ** (1/d)
        scaled_pos = self.scale_factor * pos / scale[batch.batch][:, None]
        if self.return_scale:
            return scaled_pos, scale
        return scaled_pos
    
        
class Canonicalization(nn.Module):
    def __init__(self, 
                 translate=ZeroCenter(), 
                 rotate=RotateByPrincipalComponents(),
                 normalize=Standardization(), 
                 scale=ScaleByGraphOrder()):
        super().__init__()
        self.translate = translate or IdentityTransformation()
        self.rotate = rotate or IdentityTransformation()
        self.normalize = normalize or IdentityTransformation()
        self.scale = scale or IdentityTransformation()
        
    def forward(self, pos, data):
        batch = make_batch(data)
        pos = self.translate(pos, batch)
        pos = self.rotate(pos, batch)
        pos = self.normalize(pos, batch)
        pos = self.scale(pos, batch)
        return pos
    
    
class CanonicalizationByStress(nn.Module):
    def __init__(self):
        super().__init__()
        self.canonicalize = Canonicalization(normalize=None, scale=RescaleByStress())
        
    def forward(self, pos, data):
        batch = make_batch(data)
        pos = self.canonicalize(pos, batch)
        return pos
    
