import aegnn
import argparse
import itertools
import logging
import os
import pandas as pd
import torch
import torch_geometric
import pytorch_lightning.metrics.functional as pl_metrics

from torch_geometric.data import Batch
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from tqdm import tqdm
from typing import List, Iterable

import numpy as np



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", default=3.0, help="radius of radius graph generation")
    parser.add_argument("--max-num-neighbors", default=32, help="max. number of neighbors in graph")
    parser.add_argument("--pooling-size", default=(10, 10))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


##################################################################################################
# Graph Generation ###############################################################################
##################################################################################################
def sample_initial_data(sample, num_events: int, radius: float, edge_attr, max_num_neighbors: int):
    data = Data(x=sample.x[:num_events], pos=sample.pos[:num_events])
    data.batch = torch.zeros(data.num_nodes, device=data.x.device)
    data.edge_index = torch_geometric.nn.radius_graph(data.pos, r=radius, max_num_neighbors=max_num_neighbors).long()
    data.edge_attr = edge_attr(data).edge_attr

    edge_counts_avg = data.edge_index.shape[1] / num_events
    logging.debug(f"Average edge counts in initial data = {edge_counts_avg}")
    return data


def create_and_run_model(dm, num_events: int, index: int, device: torch.device, args: argparse.Namespace, **kwargs):
    edge_attr = torch_geometric.transforms.Cartesian(cat=False, max_value=10.0)
    dataset = dm.train_dataset
    assert dm.shuffle is False  # ensuring same samples over experiments

    # Sample initial data of certain length from dataset sample. Sample num_events samples from one
    # dataset, and create the subsequent event as the one to be added.
    sample = dataset[index % len(dataset)]
    sample.pos = sample.pos[:, :2]
    events_initial = sample_initial_data(sample, num_events, args.radius, edge_attr, args.max_num_neighbors)

    index_new = min(num_events, sample.num_nodes - 1)
    x_new = sample.x[index_new, :].view(1, -1)
    pos_new = sample.pos[index_new, :2].view(1, -1)
    event_new = Data(x=x_new, pos=pos_new, batch=torch.zeros(1, dtype=torch.long))

    # Initialize model and make it asynchronous (recognition model, so num_outputs = num_classes of input dataset).
    input_shape = torch.tensor([*dm.dims, events_initial.pos.shape[-1]], device=device)
    model = aegnn.models.networks.GraphRes(dm.name, input_shape, dm.num_classes, pooling_size=args.pooling_size)
    model.to(device)
    model = aegnn.asyncronous.make_model_asynchronous(model, args.radius, list(dm.dims), edge_attr, **kwargs)

    # Run experiment, i.e. initialize the asynchronous graph and iteratively add events to it.
    _ = model.forward(events_initial.to(device))  # initialization
    _ = model.forward(event_new.to(device))
    del events_initial, event_new
    return model


##################################################################################################
# Logging ########################################################################################
##################################################################################################
def get_log_values(model, attr: str, log_key: str, **log_dict):
    """"Get log values for the given attribute key, both for each layer and in total and for the dense and sparse
    update. Thereby, approximate the logging for the dense with the data for the initial update as
    count(initial events) >>> count(new events)
    """
    assert hasattr(model, attr)
    log_values = []
    for layer, nn in model._modules.items():
        if hasattr(nn, attr):
            logs = getattr(nn, attr)
            log_values.append({"layer": layer, log_key: logs[0], "model": "gnn_dense", **log_dict})
            for log_i in logs[1:]:
                log_values.append({"layer": layer, log_key: log_i, "model": "ours", **log_dict})

    logs = getattr(model, attr)
    log_values.append({"model": "gnn_dense", log_key: logs[0], "layer": "total", **log_dict})
    for log_i in logs[1:]:
        log_values.append({"model": "ours", log_key: log_i, "layer": "total", **log_dict})

    return log_values


##################################################################################################
# Experiments ####################################################################################
##################################################################################################
def run_experiments(dm, args, experiments: List[int], num_trials: int, device: torch.device, **model_kwargs
                    ) -> pd.DataFrame:
    results_df = pd.DataFrame()
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "aegnn_results", "flops.pkl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    runs = list(itertools.product(experiments, list(range(num_trials))))
    for num_events, exp_id in tqdm(runs):
        model = create_and_run_model(dm, num_events, index=exp_id, args=args, device=device, **model_kwargs)

        # Get the logged flops and timings, both layer-wise and in total.
        results_flops = get_log_values(model, attr="asy_flops_log", log_key="flops", num_events=num_events)
        results_runtime = get_log_values(model, attr="asy_runtime_log", log_key="runtime", num_events=num_events)
        results_df = results_df.append(results_flops + results_runtime, ignore_index=True)
        results_df.to_pickle(output_file)

        # Fully reset run to ensure independence between subsequent experiments.
        del model  # fully delete model
        torch.cuda.empty_cache()  # clear memory

    print(f"Results are logged in {output_file}")
    return results_df


# -------------------

def sample_batch(batch_idx: torch.Tensor, num_samples: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Samples a subset of graphs in a batch and returns the sampled nodes and batch_idx.

    >> batch = torch.from_numpy(np.random.random_integers(0, 10, size=100))
    >> batch = torch.sort(batch).values
    >> sample_batch(batch, max_num_events=2)
    """
    subset = []
    subset_batch_idx = []
    for i in torch.unique(batch_idx):
        batch_idx_i = torch.nonzero(torch.eq(batch_idx, i)).flatten()
        # sample_idx_i = torch.randperm(batch_idx_i.numel())[:num_samples]
        # subset.append(batch_idx_i[sample_idx_i])
        sample_idx_i = batch_idx_i[:num_samples]
        subset.append(sample_idx_i)
        subset_batch_idx.append(torch.ones(sample_idx_i.numel()) * i)
    return torch.cat(subset).long(), torch.cat(subset_batch_idx).long()



def evaluate(model, args, data_module: Iterable[Batch], max_num_events: int) -> float:
    accuracy = []
    data_loader = data_module.test_dataloader(num_workers=16).__iter__()

    for i, batch in enumerate(data_loader):
        batch_idx = getattr(batch, 'batch')
        subset, subset_batch_idx = sample_batch(batch_idx, num_samples=max_num_events)
        is_in_subset = torch.zeros(batch_idx.numel(), dtype=torch.bool)
        is_in_subset[subset] = True

        edge_index, edge_attr = subgraph(is_in_subset, batch.edge_index, edge_attr=batch.edge_attr, relabel_nodes=True)
        sample = Batch(x=batch.x[is_in_subset, :], pos=batch.pos[is_in_subset, :], y=batch.y,
                       edge_index=edge_index, edge_attr=edge_attr, batch=subset_batch_idx)
        logging.debug(f"Done data-processing, resulting in {sample}")

        # Make model async based on batch edge attrb
        model = aegnn.asyncronous.make_model_asynchronous(model, args.radius, list(dm.dims), edge_attr)

        sample = sample.to(model.device)
        outputs_i = model.forward(sample)
        y_hat_i = torch.argmax(outputs_i, dim=-1)

        accuracy_i = pl_metrics.accuracy(preds=y_hat_i, target=sample.y).cpu().numpy()
        accuracy.append(accuracy_i)
    return float(np.mean(accuracy))


def main(args, model, data_module):
    event_counts = [1000]
    df = pd.DataFrame()

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "aegnn_results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "accuracy_per_events.pkl")

    for count in tqdm(event_counts):
        
        accuracy = evaluate(model, args, data_module, max_num_events=count)
        logging.debug(f"Evaluation with max_num_events = {count} => Recognition accuracy = {accuracy}")

        df = df.append({"accuracy": accuracy, "max_num_events": count}, ignore_index=True)
        df.to_pickle(output_file)

    print(f"Results are logged in {output_file}")
    return df


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        _ = aegnn.utils.loggers.LoggingLogger(None, name="debug")

    model_eval = torch.load(args.model_file).to(args.device)
    dm = aegnn.datasets.by_name(args.dataset).from_argparse_args(args)
    dm.setup()

    main(args, model_eval, dm)