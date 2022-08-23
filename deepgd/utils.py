from ._dependencies import *

from .functions import *
from .modules import *
from .transform import *
from .metrics import *
from ipynb.fs.defs.losses import *


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
    #     print(f'Iteration finished. {failed_count} out of {len(loader)} failed!')
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


def preprocess_batch(model, batch):
    if not isinstance(batch, Batch):
        batch = Batch.from_data_list([batch])
    device = next(model.parameters()).device
    return batch.to(device)


def to_batch(batch, device='cpu'):
    if not isinstance(batch, Batch):
        batch = Batch.from_data_list([batch])
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


def evaluate(batch, pred, gt):
    canonicalize = CanonicalizationByStress()
    stress_criterion = Stress()
    xing_criterion = Xing()
    xangle_criterion = XingAngle()
    rxa_criterion = RealXingAngle()
    l1angle_criterion = L1AngularLoss()
    edge_criterion = FixedMeanEdgeLengthVarianceLoss()
    ring_criterion = ExponentialRingLoss()
    tsne_criterion = TSNELoss()
    spc_criterion = SPC(reduce=None)
    
    pred = canonicalize(pred, batch)
    gt = canonicalize(gt, batch)

    gt_stress = stress_criterion(gt, batch)
    gt_xing = xing_criterion(gt, batch)
    gt_xangle = xangle_criterion(gt, batch)
    gt_rxa = rxa_criterion(gt, batch)
    gt_l1angle = l1angle_criterion(gt, batch)
    gt_edge = edge_criterion(gt, batch)
    gt_ring = ring_criterion(gt, batch)
    gt_tsne = tsne_criterion(gt, batch)

    stress = stress_criterion(pred, batch)
    xing = xing_criterion(pred, batch)
    xangle = xangle_criterion(pred, batch)
    rxa = rxa_criterion(pred, batch)
    l1angle = l1angle_criterion(pred, batch)
    edge = edge_criterion(pred, batch)
    ring = ring_criterion(pred, batch)
    tsne = tsne_criterion(pred, batch)

    stress_spc = spc_criterion(stress, gt_stress)
    xing_spc = spc_criterion(xing, gt_xing)
    xangle_spc = spc_criterion(xangle, gt_xangle)
    rxa_spc = spc_criterion(rxa, gt_rxa)
    l1angle_spc = spc_criterion(l1angle, gt_l1angle)
    edge_spc = spc_criterion(edge, gt_edge)
    ring_spc = spc_criterion(ring, gt_ring)
    tsne_spc = spc_criterion(tsne, gt_tsne)
        
    return {
        'stress': stress.item(),
        'gt_stress': gt_stress.item(),
        'stress_spc': stress_spc.item(),
        'xing': xing.item(),
        'gt_xing': gt_xing.item(),
        'xing_spc': xing_spc.item(),
        'xangle': xangle.item(),
        'gt_xangle': gt_xangle.item(),
        'xangle_spc': xangle_spc.item(),
        'rxa': rxa.item(),
        'gt_rxa': gt_rxa.item(),
        'rxa_spc': rxa_spc.item(),
        'l1angle': l1angle.item(),
        'gt_l1angle': gt_l1angle.item(),
        'l1angle_spc': l1angle_spc.item(),
        'edge': edge.item(),
        'gt_edge': gt_edge.item(),
        'edge_spc': edge_spc.item(),
        'ring': ring.item(),
        'gt_ring': gt_ring.item(),
        'ring_spc': ring_spc.item(),
        'tsne': tsne.item(),
        'gt_tsne': gt_tsne.item(),
        'tsne_spc': tsne_spc.item(),
    }


def test(model, dataset, idx_range, callback=None, pred_list=None, gt_list=None, **model_params):
    if callback is None:
        callback = lambda *_, **__: None

    device = next(iter(model.parameters())).device

    stress = []
    stress_spc = []
    gt_stress = []
    xing = []
    xing_spc = []
    gt_xing = []
    xangle = []
    xangle_spc = []
    gt_xangle = []
    rxa = []
    rxa_spc = []
    gt_rxa = []
    l1angle = []
    l1angle_spc = []
    gt_l1angle = []
    edge = []
    edge_spc = []
    gt_edge = []
    ring = []
    ring_spc = []
    gt_ring = []
    tsne = []
    tsne_spc = []
    gt_tsne = []
    
    model.eval()
    for idx in tqdm(idx_range):
        with torch.no_grad():
            batch = to_batch(dataset[idx], device)
            pred = torch.tensor(pred_list[idx]).float().to(device) if pred_list is not None else model(batch, **model_params)
            gt = torch.tensor(gt_list[idx]).float().to(device) if gt_list is not None else batch.gt_pos
            metrics = evaluate(batch, pred, gt)

        stress.append(metrics['stress'])
        stress_spc.append(metrics['stress_spc'])
        gt_stress.append(metrics['gt_stress'])
        xing.append(metrics['xing'])
        xing_spc.append(metrics['xing_spc'])
        gt_xing.append(metrics['gt_xing'])
        xangle.append(metrics['xangle'])
        xangle_spc.append(metrics['xangle_spc'])
        gt_xangle.append(metrics['gt_xangle'])
        rxa.append(metrics['rxa'])
        rxa_spc.append(metrics['rxa_spc'])
        gt_rxa.append(metrics['gt_rxa'])
        l1angle.append(metrics['l1angle'])
        l1angle_spc.append(metrics['l1angle_spc'])
        gt_l1angle.append(metrics['gt_l1angle'])
        edge.append(metrics['edge'])
        edge_spc.append(metrics['edge_spc'])
        gt_edge.append(metrics['gt_edge'])
        ring.append(metrics['ring'])
        ring_spc.append(metrics['ring_spc'])
        gt_ring.append(metrics['gt_ring'])
        tsne.append(metrics['tsne'])
        tsne_spc.append(metrics['tsne_spc'])
        gt_tsne.append(metrics['gt_tsne'])

        callback(idx=idx, pred=pred, metrics=metrics)

    return {
        "stress": torch.tensor(stress),
        "gt_stress": torch.tensor(gt_stress),
        "stress_spc": torch.tensor(stress_spc),
        "xing": torch.tensor(xing),
        "gt_xing": torch.tensor(gt_xing),
        "xing_spc": torch.tensor(xing_spc),
        "xangle": torch.tensor(xangle),
        "gt_xangle": torch.tensor(gt_xangle),
        "xangle_spc": torch.tensor(xangle_spc),
        "rxa": torch.tensor(rxa),
        "gt_rxa": torch.tensor(gt_rxa),
        "rxa_spc": torch.tensor(rxa_spc),
        "l1angle": torch.tensor(l1angle),
        "gt_l1angle": torch.tensor(gt_l1angle),
        "l1angle_spc": torch.tensor(l1angle_spc),
        "edge": torch.tensor(edge),
        "gt_edge": torch.tensor(gt_edge),
        "edge_spc": torch.tensor(edge_spc),
        "ring": torch.tensor(ring),
        "gt_ring": torch.tensor(gt_ring),
        "ring_spc": torch.tensor(ring_spc),
        "tsne": torch.tensor(tsne),
        "gt_tsne": torch.tensor(gt_tsne),
        "tsne_spc": torch.tensor(tsne_spc),
    }


def get_performance_metrics(model, data, idx, criteria_list=None, eval_method=None, pred_pos=None, gt_pos=None, **model_params):
    with torch.no_grad():
        model.eval()
        canonicalize = CanonicalizationByStress()

        data = preprocess_batch(model, data)
        stress_criterion = Stress()
        xing_criterion = Xing()
        xangle_criterion = XingAngle()
        l1angle_criterion = L1AngularLoss()
        edge_criterion = FixedMeanEdgeLengthVarianceLoss()
        ring_criterion = ExponentialRingLoss()
        tsne_criterion = TSNELoss()
        
        spc_criterion = SPC(reduce=None)

        #         if gt_stress is None:
        #             gt = get_ground_truth(data)
        #             gt_stress = stress_criterion(gt, data)

        # gt_stress = load_ground_truth(idx, 'stress', gt_file)
        # gt_xing = load_ground_truth(idx, 'xing', gt_file)
        # gt_l1_angle = load_ground_truth(idx, 'l1_angle', gt_file)
        # gt_edge = load_ground_truth(idx, 'edge', gt_file)
        # gt_ring = load_ground_truth(idx, 'ring', gt_file)
        # gt_tsne = load_ground_truth(idx, 'tsne', gt_file)

        if gt_pos is None:
            gt_pos = data.gt_pos
        gt_pos = canonicalize(gt_pos, data)

        gt_stress = stress_criterion(gt_pos, data)
        gt_xing = xing_criterion(gt_pos, data)
        gt_xangle = xangle_criterion(gt_pos, data)
        gt_l1angle = l1angle_criterion(gt_pos, data)
        gt_edge = edge_criterion(gt_pos, data)
        gt_ring = ring_criterion(gt_pos, data)
        gt_tsne = tsne_criterion(gt_pos, data)

        if eval_method is None:
            raw_pred = model(data, **model_params)
            pred = canonicalize(raw_pred, data)
        elif eval_method == "tsne":
            hidden = model(data, output_hidden=True, numpy=True, **model_params)
            raw_pred = torch.tensor(tsne_project(hidden[-2]))
            pred = canonicalize(raw_pred, data)
        elif eval_method == "umap":
            hidden = model(data, output_hidden=True, numpy=True, **model_params)
            raw_pred = torch.tensor(umap_project(hidden[-2]))
            pred = canonicalize(raw_pred, data)
        elif eval_method == "gt":
            raw_pred = data.gt_pos
            pred = canonicalize(raw_pred, data)
        elif eval_method == "load":
            raw_pred = pred_pos
            pred = canonicalize(raw_pred, data)

        stress = stress_criterion(pred, data)
        xing = xing_criterion(pred, data)
        xangle = xangle_criterion(pred, data)
        l1angle = l1angle_criterion(pred, data)
        edge = edge_criterion(pred, data)
        ring = ring_criterion(pred, data)
        tsne = tsne_criterion(pred, data)

        stress_spc = spc_criterion(stress, gt_stress)
        xing_spc = spc_criterion(xing, gt_xing)
        xangle_spc = spc_criterion(xangle, gt_xangle)
        l1angle_spc = spc_criterion(l1angle, gt_l1angle)
        edge_spc = spc_criterion(edge, gt_edge)
        ring_spc = spc_criterion(ring, gt_ring)
        tsne_spc = spc_criterion(tsne, gt_tsne)

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
        'gt_stress': gt_stress.item(),
        'stress_spc': stress_spc.item(),
        'xing': xing.item(),
        'gt_xing': gt_xing.item(),
        'xing_spc': xing_spc.item(),
        'xangle': xangle.item(),
        'gt_xangle': gt_xangle.item(),
        'xangle_spc': xangle_spc.item(),
        'l1angle': l1angle.item(),
        'gt_l1angle': gt_l1angle.item(),
        'l1angle_spc': l1angle_spc.item(),
        'edge': edge.item(),
        'gt_edge': gt_edge.item(),
        'edge_spc': edge_spc.item(),
        'ring': ring.item(),
        'gt_ring': gt_ring.item(),
        'ring_spc': ring_spc.item(),
        'tsne': tsne.item(),
        'gt_tsne': gt_tsne.item(),
        'tsne_spc': tsne_spc.item(),
        'resolution_score': resolution_score.item(),
        'min_angle': min_angle.item(),
        'losses': list(map(torch.Tensor.item, losses))
    }


def test_old(model, criteria_list, dataset, idx_range, callback=None, eval_method=None, gt_pos=None, pred_pos=None, **model_params):
    if callback is None:
        callback = lambda *_, **__: None

    device = next(iter(model.parameters())).device

    stress = []
    stress_spc = []
    xing = []
    xing_spc = []
    xangle = []
    xangle_spc = []
    l1angle = []
    l1angle_spc = []
    edge = []
    edge_spc = []
    ring = []
    ring_spc = []
    tsne = []
    tsne_spc = []
    resolution_score = []
    min_angle = []
    losses = []
    for idx in tqdm(idx_range):
        g_pos = torch.tensor(gt_pos[idx]).float().to(device) if gt_pos is not None else None
        p_pos = torch.tensor(pred_pos[idx]).float().to(device) if pred_pos is not None else None
        pred, metrics = get_performance_metrics(model, dataset[idx], idx,
                                                criteria_list=criteria_list,
                                                eval_method=eval_method,
                                                pred_pos=p_pos,
                                                gt_pos=g_pos,
                                                **model_params)

        stress.append(metrics['stress'])
        stress_spc.append(metrics['stress_spc'])
        xing.append(metrics['xing'])
        xing_spc.append(metrics['xing_spc'])
        xangle.append(metrics['xangle'])
        xangle_spc.append(metrics['xangle_spc'])
        l1angle.append(metrics['l1angle'])
        l1angle_spc.append(metrics['l1angle_spc'])
        edge.append(metrics['edge'])
        edge_spc.append(metrics['edge_spc'])
        ring.append(metrics['ring'])
        ring_spc.append(metrics['ring_spc'])
        tsne.append(metrics['tsne'])
        tsne_spc.append(metrics['tsne_spc'])
        resolution_score.append(metrics['resolution_score'])
        min_angle.append(metrics['min_angle'])
        losses.append(metrics['losses'])

        callback(idx=idx, pred=pred, metrics=metrics)

    return {
        "stress": torch.tensor(stress),
        "stress_spc": torch.tensor(stress_spc),
        "xing": torch.tensor(xing),
        "xing_spc": torch.tensor(xing_spc),
        "xangle": torch.tensor(xangle),
        "xangle_spc": torch.tensor(xangle_spc),
        "l1angle": torch.tensor(l1angle),
        "l1angle_spc": torch.tensor(l1angle_spc),
        "edge": torch.tensor(edge),
        "edge_spc": torch.tensor(edge_spc),
        "ring": torch.tensor(ring),
        "ring_spc": torch.tensor(ring_spc),
        "tsne": torch.tensor(tsne),
        "tsne_spc": torch.tensor(tsne_spc),
        "resolution_score": torch.tensor(resolution_score),
        "min_angle": torch.tensor(min_angle),
        "losses": torch.tensor(losses),
    }


def get_gt_performance_metrics(data, G=None, gt_stress=None, criteria_list=None, **model_params):
    if not isinstance(batch, Batch):
        data = Batch.from_data_list([data])

    stress_criterion = Stress()

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


def graph_vis(G, pos, *, highlight_edge=None, file_name=None, **kwargs):
    G = nx.Graph(G)
    graph_attr = dict(node_size=100,
                      with_labels=False,
                      labels=dict(zip(list(G.nodes), map(lambda n: n if type(n) is int else n[1:], list(G.nodes)))),
                      font_color="white",
                      font_weight="bold",
                      font_size=12)
    graph_attr.update(kwargs)
    if highlight_edge is not None:
        edges = list(G.edges)
        attrs = {edges[e]: {'c': 'r', 'w': 2} for e in np.unique(highlight_edge)}
        nx.set_edge_attributes(G, attrs)
        colors = [G[u][v]['c'] if 'c' in G[u][v] else 'black' for u,v in edges]
        weights = [G[u][v]['w'] if 'w' in G[u][v] else 1 for u,v in edges]
        graph_attr.update(dict(edge_color=colors, width=weights))
    for i, (n, p) in enumerate(pos):
        G.nodes[i]['pos'] = n, p
    pos = nx.get_node_attributes(G, name='pos')
    if file_name is not None:
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