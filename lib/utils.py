from .imports import *
from .functions import *
from .modules import *
from ipynb.fs.defs.losses import *


# TODO: setitem setattr
class Config:
    class Store:
        def __init__(self, data: dict):
            def wrap(kvpair):
                key, value = kvpair
                if type(value) is dict:
                    return key, Config.Store(data=value)
                return kvpair
            self.__dict__ = dict(map(wrap, data.items()))

        def __getitem__(self, item):
            def unwrap(kvpair):
                key, value = kvpair
                if type(value) is Config.Store:
                    return key, value[...]
                return kvpair
            if item is ...:
                return dict(map(unwrap, self.__dict__.items()))
            return self.__dict__[item]

        def __repr__(self):
            return pformat(self.__dict__)#, sort_dicts=False)

        def __str__(self):
            return str(self.__dict__)
        
    def __init__(self, file):
        self.file = file
        
    def __getitem__(self, item):
        data = Config.Store(json.load(open(self.file)))
        if item is None:
            return data
        return data[item]
    
    def __getattr__(self, attr):
        return self[attr]
    
    def __repr__(self):
        return pformat(self[None])#, sort_dicts=False)
    
    def __str__(self):
        return str(self[None])
    

class StaticConfig:
    class Store:
        def __init__(self, data: dict):
            def wrap(kvpair):
                key, value = kvpair
                if type(value) is dict:
                    return key, Config.Store(data=value)
                return kvpair
            self.__dict__ = dict(map(wrap, data.items()))

        def __getitem__(self, item):
            def unwrap(kvpair):
                key, value = kvpair
                if type(value) is Config.Store:
                    return key, value[...]
                return kvpair
            if item is ...:
                return dict(map(unwrap, self.__dict__.items()))
            return self.__dict__[item]

        def __repr__(self):
            return pformat(self.__dict__)#, sort_dicts=False)

        def __str__(self):
            return str(self.__dict__)
        
    def __init__(self, data):
        self.data = Config.Store(data)
        
    def __getitem__(self, item):
        if item is None:
            return self.data
        return self.data[item]
    
    def __getattr__(self, attr):
        return self[attr]
    
    def __repr__(self):
        return pformat(self[None])#, sort_dicts=False)
    
    def __str__(self):
        return str(self[None])
    

def generate_polygon(n, radius=1):
    node_pos = [(radius * np.cos(2 * np.pi * i / n),
                 radius * np.sin(2 * np.pi * i / n)) for i in range(n)]
    x = torch.tensor(node_pos, dtype=torch.float)
    return x


def find_intersect(segments, accurate=True):
    intersect(segments)
    

def cuda_memsafe_iter(loader, callback):
    results = []
    batch_iter = iter(loader)
    batch = None
    failed_count = 0
    while True:
        try:
            batch = next(batch_iter)
            torch.cuda.empty_cache()
            results.append(callback(batch=batch))
        except StopIteration:
            break
        except RuntimeError:
            failed_count += 1
            print('CUDA memory overflow! Skip batch...')
            del batch
            torch.cuda.empty_cache()
    print(f'Iteration finished. {failed_count} out of {len(loader)} failed!')
    return results

    
def train(model, criterion, optimizer, data_loader, callback=None):
    if callback is None:
        callback = lambda *_, **__: None
    model.train()
    def train_one_batch(batch):
        optimizer.zero_grad()
        output, loss, components = predict(model, batch, criterion)
        loss.backward()
        optimizer.step()
        callback(output=output, loss=loss, components=components)
        return loss.item(), [comp.item() for comp in components]
    results = cuda_memsafe_iter(data_loader, train_one_batch)
    loss_all, components_all = zip(*results)
    return np.mean(loss_all), np.mean(components_all, axis=0)


def train_with_dynamic_weights(model, controller, criterion, optimizer, data_loader, callback=None):
    if callback is None:
        callback = lambda *_, **__: None
    model.train()
    loss_all, components_all = [], []
    for batch in data_loader:
        torch.cuda.empty_cache()
        gamma = F.normalize(torch.rand(controller.n_criteria), p=1, dim=0)
        controller.set_gamma(gamma.tolist())
        controller.step()
        optimizer.zero_grad()
        output, loss, components = predict(model, batch, criterion, weights=gamma.to(next(model.parameters()).device))
        loss.backward()
        optimizer.step()
        loss_all.append(loss.item())
        components_all.append([comp.item() for comp in components])
        callback(output=output, loss=loss, components=components)
    return np.mean(loss_all), np.mean(components_all, axis=0)
    

def validate(model, criterion, data_loader, callback=None):
    if callback is None:
        callback = lambda *_, **__: None
    with torch.no_grad():
        model.eval()
        def val_one_batch(batch):
            output, loss, components = predict(model, batch, criterion)
            callback(output=output, loss=loss, components=components)
            return loss.item(), [comp.item() for comp in components]
        results = cuda_memsafe_iter(data_loader, val_one_batch)
        loss_all, components_all = zip(*results)
    return np.mean(loss_all), np.mean(components_all, axis=0)


def validate_with_dynamic_weights(model, controller, criterion, data_loader, callback=None):
    if callback is None:
        callback = lambda *_, **__: None
    with torch.no_grad():
        model.eval()
        loss_all, components_all = [], []
        for batch in data_loader:
            torch.cuda.empty_cache()
            gamma = F.normalize(torch.rand(controller.n_criteria), p=1, dim=0)
            controller.set_gamma(gamma.tolist())
            controller.step()
            output, loss, components = predict(model, batch, criterion, weights=gamma.to(next(model.parameters()).device))
            loss_all.append(loss.item())
            components_all.append([comp.item() for comp in components])
            callback(output=output, loss=loss, components=components)
    return np.mean(loss_all), np.mean(components_all, axis=0)


def test(model, criteria_list, dataset, idx_range, callback=None, eval_method=None, **model_params):
    if callback is None:
        callback = lambda *_, **__: None
    stress = []
    stress_spc = []
    l1_angle = []
    l1_angle_spc = []
    edge = []
    edge_spc = []
    ring = []
    ring_spc = []
    tsne = []
    resolution_score = []
    min_angle = []
    losses = []
    for idx in tqdm(idx_range):
        gt_stress = load_ground_truth_stress(idx)
        pred, metrics = get_performance_metrics(model, dataset[idx], idx,
                                                criteria_list=criteria_list,
                                                eval_method=eval_method,
                                                **model_params)

        stress.append(metrics['stress'])
        stress_spc.append(metrics['stress_spc'])
        l1_angle.append(metrics['l1_angle'])
        l1_angle_spc.append(metrics['l1_angle_spc'])
        edge.append(metrics['edge'])
        edge_spc.append(metrics['edge_spc'])
        ring.append(metrics['ring'])
        ring_spc.append(metrics['ring_spc'])
        tsne.append(metrics['tsne'])
        resolution_score.append(metrics['resolution_score'])
        min_angle.append(metrics['min_angle'])
        losses.append(metrics['losses'])

        callback(idx=idx, pred=pred, metrics=metrics)
    
    return {
        "stress": torch.tensor(stress),
        "stress_spc": torch.tensor(stress_spc),
        "l1_angle": torch.tensor(l1_angle),
        "l1_angle_spc": torch.tensor(l1_angle_spc),
        "edge": torch.tensor(edge),
        "edge_spc": torch.tensor(edge_spc),
        "ring": torch.tensor(ring),
        "ring_spc": torch.tensor(ring_spc),
        "tsne": torch.tensor(tsne),
        "resolution_score": torch.tensor(resolution_score),
        "min_angle": torch.tensor(min_angle),
        "losses": torch.tensor(losses),
    }


def preprocess_batch(model, batch):
    if type(batch) is not Batch:
        batch = Batch.from_data_list([batch])
    device = next(model.parameters()).device
    return batch.to(device)


def predict(model, batch, criterion, **model_params):
    if type(criterion) is not CompositeLoss:
        criterion = CompositeLoss([criterion])
    batch = preprocess_batch(model, batch)
    output = model(batch, **model_params)
    pred = output[0] if output is tuple else output
    loss, components = criterion(pred, batch, return_components=True)
    return output, loss, components
    
    
@lru_cache(maxsize=1)
def load_gt_dataframe(file="gt.csv"):
    return pd.read_csv(file)
    
    
def load_ground_truth(idx, metric, file="gt.csv"):
    gt = load_gt_dataframe(file)
    return gt[gt['index'] == idx][metric].to_numpy()[0]
    

def load_ground_truth_stress(index, file='scaled_gt_loss.csv'):
    gt_losses = pd.read_csv(file).to_numpy()
    if (gt_losses[:, 0] == index).any():
        return gt_losses[gt_losses[:, 0] == index][0, 1]
    return None
    
    
def get_performance_metrics(model, data, idx, criteria_list=None, eval_method=None, **model_params):
    with torch.no_grad():
        model.eval()
        data = preprocess_batch(model, data)
        stress_criterion = NormalizedStressLoss()
        l1_angle_criterion = L1AngularLoss()
        edge_criterion = FixedMeanEdgeLengthVarianceLoss()
        ring_criterion = ExponentialRingLoss()
        tsne_criterion = TSNELoss()
        
#         if gt_stress is None:
#             gt = get_ground_truth(data)
#             gt_stress = stress_criterion(gt, data)
        
        gt_stress = load_ground_truth(idx, 'stress')
        gt_l1_angle = load_ground_truth(idx, 'l1_angle')
        gt_edge = load_ground_truth(idx, 'edge')
        gt_ring = load_ground_truth(idx, 'ring')
    
        if eval_method is None:
            raw_pred = model(data, **model_params)
            pred = rescale_with_minimized_stress(raw_pred, data)
        elif eval_method == "tsne":
            hidden = model(data, output_hidden=True, numpy=True, **model_params)
            raw_pred = torch.tensor(tsne_project(hidden[-2]))
            pred = rescale_with_minimized_stress(raw_pred, data)
        elif eval_method == "umap":
            hidden = model(data, output_hidden=True, numpy=True, **model_params)
            raw_pred = torch.tensor(umap_project(hidden[-2]))
            pred = rescale_with_minimized_stress(raw_pred, data)
        
        stress = stress_criterion(pred, data)
        l1_angle = l1_angle_criterion(pred, data)
        edge = edge_criterion(pred, data)
        ring = ring_criterion(pred, data)
        
        stress_spc = (stress - gt_stress) / np.maximum(gt_stress, stress.cpu().numpy())
        l1_angle_spc = (l1_angle - gt_l1_angle) / np.maximum(gt_l1_angle, l1_angle.cpu().numpy())
        edge_spc = (edge - gt_edge) / np.maximum(gt_edge, edge.cpu().numpy())
        ring_spc = (ring - gt_ring) / np.maximum(gt_ring, ring.cpu().numpy())
    
        tsne = tsne_criterion(pred, data)
        theta, degree, node = get_radians(pred, data, 
                                          return_node_degrees=True,
                                          return_node_indices=True)
        resolution_score = get_resolution_score(theta, degree, node)
        min_angle = get_min_angle(theta)
        if criteria_list is not None:
            other_criteria = CompositeLoss(criteria_list)
            _, losses = other_criteria(pred, data, return_components=True)

    return pred.cpu().numpy(), {
        'stress': stress.item(),
        'stress_spc': stress_spc.item(),
        'l1_angle': l1_angle.item(),
        'l1_angle_spc': l1_angle_spc.item(),
        'edge': edge.item(),
        'edge_spc': edge_spc.item(),
        'ring': ring.item(),
        'ring_spc': ring_spc.item(),
        'tsne': tsne.item(),
        'resolution_score': resolution_score.item(),
        'min_angle': min_angle.item(),
        'losses': list(map(torch.Tensor.item, losses))
    }


def get_gt_performance_metrics(data, G=None, gt_stress=None, criteria_list=None, **model_params):
    if type(data) is not Batch:
        data = Batch.from_data_list([data])
        
    stress_criterion = StressLoss()

    if gt_stress is None:
        gt = get_ground_truth(data, G)
        gt_stress = stress_criterion(gt, data)

    if criteria_list is not None:
        other_criteria = CompositeLoss(criteria_list)
        _, gt_losses = other_criteria(gt, data, return_components=True)

    
    return {
        'gt_stress': gt_stress.item(),
        'gt_losses': list(map(torch.Tensor.item, gt_losses))
    }
    
    
def graph_vis(G, node_pos, file_name=None, **kwargs):
    graph_attr = dict(node_size=100, 
                      with_labels=False, 
                      labels=dict(zip(list(G.nodes), map(lambda n: n if type(n) is int else n[1:], list(G.nodes)))),
                      font_color="white", 
                      font_weight="bold",
                      font_size=12)
    graph_attr.update(kwargs)
    for i, (n, p) in enumerate(node_pos):
        G.nodes[i]['pos'] = n, p
    pos = nx.get_node_attributes(G, name='pos')
    plt.figure()
    nx.draw(G, pos, **graph_attr)
    plt.axis('equal')
    if file_name is not None:
        plt.savefig(file_name)
        

def visualize_graph(data, file_name=None, **kwargs):
    G = tg_to_nx(data)
    graph_attr = dict(node_size=100, 
                      with_labels=False, 
                      labels=dict(zip(list(G.nodes), map(lambda n: str(n), list(G.nodes)))),
                      font_color="white", 
                      font_weight="bold",
                      font_size=12)
    graph_attr.update(kwargs)
    pos = nx.get_node_attributes(G, name='pos')
    plt.figure()
    nx.draw(G, pos, **graph_attr)
    if file_name is not None:
        plt.savefig(file_name)
    
    
def pca_project(vector, n=2):
    if vector.shape[1] == n:
        return vector
    pca = PCA(n_components=n)
    return pca.fit_transform(vector)


def tsne_project(vector, n=2):
    tsne = TSNE(n_components=n)
    return tsne.fit_transform(vector)


def umap_project(vector, n=2, **kwargs):
    umap = UMAP(n_components=n, **kwargs)
    return umap.fit_transform(vector)