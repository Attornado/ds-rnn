"""Graph-theoretic metrics for sparsity-mask analysis."""

from typing import Dict

import torch


def shannon_entropy_entries(adj: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy of all edge weights treated as one distribution."""
    flat = adj.flatten(start_dim=-2)
    p = flat / (flat.sum(dim=-1, keepdim=True) + eps)
    return -(p * (p + eps).log()).sum(dim=-1)


def degree_distribution_entropy(adj: torch.Tensor, mode: str = "out", eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy of the (out-)degree distribution over nodes."""
    k = adj.sum(dim=-1 if mode == "out" else -2)
    p = k / (k.sum(dim=-1, keepdim=True) + eps)
    return -(p * (p + eps).log()).sum(dim=-1)


def random_walk_entropy(adj: torch.Tensor, num_iter: int = 50, tol: float = 1e-6) -> torch.Tensor:
    """Entropy rate of the natural random walk on a weighted directed graph."""
    d_out = adj.sum(dim=-1, keepdim=True)
    p = adj / (d_out + 1e-12)

    shape = adj.shape[:-2]
    n = adj.size(-1)
    pi = torch.full((*shape, n), 1.0 / n, device=adj.device, dtype=adj.dtype)
    for _ in range(num_iter):
        pi_next = torch.matmul(pi.unsqueeze(-2), p).squeeze(-2)
        if torch.max(torch.abs(pi_next - pi)).item() < tol:
            pi = pi_next
            break
        pi = pi_next

    p_safe = p.clamp(min=1e-12)
    h_local = -(p_safe * torch.log(p_safe)).sum(dim=-1)
    return (pi * h_local).sum(dim=-1)


def von_neumann_entropy(adj: torch.Tensor, mode: str = "all", eps: float = 1e-12) -> torch.Tensor:
    """Von Neumann entropy via the (symmetrized) Laplacian density matrix."""
    if mode in ["all", "sym"]:
        adj_sym = 0.5 * (adj + adj.transpose(-2, -1))
        k = adj_sym.sum(dim=-1)
    elif mode == "in":
        adj_sym = adj
        k = adj_sym.sum(dim=-2)
    elif mode == "out":
        adj_sym = adj
        k = adj_sym.sum(dim=-1)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    deg = torch.diag_embed(k)
    lapl = deg - adj_sym
    if mode in ["in", "out"]:
        lapl = 0.5 * (lapl + lapl.transpose(-2, -1))

    eig = torch.linalg.eigvalsh(lapl)
    eig = eig.clamp(min=0)
    eig = eig / (eig.sum(dim=1, keepdim=True) + eps)
    return -(eig * (eig + eps).log()).sum(dim=-1)


def topological_entropy(adj: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Log spectral radius of the adjacency matrix."""
    spectral_radius = torch.linalg.eigvalsh(adj).abs().max(dim=-1).values
    return torch.log(spectral_radius + eps)


def graph_spectral_gap(adj: torch.Tensor, mode: str = "out", eps: float = 1e-12) -> torch.Tensor:
    """Laplacian spectral gap (difference of the two smallest Laplacian eigenvalues)."""
    if mode == "in":
        k = adj.sum(dim=-2)
    elif mode == "out":
        k = adj.sum(dim=-1)
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    deg = torch.diag_embed(k)
    lapl = deg - adj
    lapl = 0.5 * (lapl + lapl.transpose(-2, -1))
    eig = torch.linalg.eigvalsh(lapl)
    top2 = eig.abs().topk(k=2).values
    return top2[..., 0] - top2[..., 1]


def _simple_metrics(adj: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Density, total weight, degree/strength, and reciprocity."""
    n = adj.shape[-1]
    is_edge = adj > 0

    density = is_edge.float().flatten(start_dim=-2).mean(dim=-1)
    total_weight = adj.flatten(start_dim=-2).sum(dim=-1)
    avg_out_degree = is_edge.float().sum(dim=-1).mean(dim=-1)
    avg_in_degree = is_edge.float().sum(dim=-2).mean(dim=-1)
    avg_out_strength = adj.sum(dim=-1).mean(dim=-1)
    avg_in_strength = adj.sum(dim=-2).mean(dim=-1)

    mutual = (is_edge & is_edge.transpose(-2, -1)).float()
    mutual_pairs = mutual.sum(dim=(-2, -1)) / 2.0
    edge_counts = is_edge.float().sum(dim=(-2, -1))
    reciprocity = 2 * mutual_pairs / edge_counts.clamp(min=1.0)

    return {
        "density": density,
        "reciprocity": reciprocity,
        "total_weight": total_weight,
        "avg_out_degree": avg_out_degree,
        "avg_in_degree": avg_in_degree,
        "avg_out_strength": avg_out_strength,
        "avg_in_strength": avg_in_strength,
    }


def _advanced_metrics(adj: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Transitivity, average shortest path, and diameter (via networkx)."""
    import networkx as nx

    flat = adj.reshape(-1, adj.shape[-2], adj.shape[-1])
    batch_dims = adj.shape[:-2]

    clustering, avg_path, diameter = [], [], []
    for a in flat:
        g = nx.from_numpy_array(a.detach().cpu().numpy(), create_using=nx.DiGraph)
        clustering.append(nx.transitivity(g))
        if not nx.is_strongly_connected(g):
            comp = max(nx.strongly_connected_components(g), key=len)
            gc = g.subgraph(comp).copy()
        else:
            gc = g
        for u, v, d in gc.edges(data=True):
            d["inv_weight"] = 1.0 / d.get("weight", 1.0)
        avg_path.append(nx.average_shortest_path_length(gc, weight="inv_weight"))
        diameter.append(nx.diameter(gc))

    def to_tensor(lst):
        return torch.tensor(lst, device=adj.device).reshape(*batch_dims)

    return {
        "clustering": to_tensor(clustering),
        "avg_shortest_path": to_tensor(avg_path),
        "diameter": to_tensor(diameter),
    }


def compute_graph_metrics(adj: torch.Tensor, advanced: bool = True) -> Dict[str, torch.Tensor]:
    """Compute the suite of graph metrics for a batch of adjacency matrices.

    Args:
        adj: Adjacency tensor of shape (..., N, N).
        advanced: Whether to also compute the networkx-based connectivity
            metrics (transitivity, shortest path, diameter).
    """
    out = {
        **_simple_metrics(adj),
        "laplacian_spectral_gap": graph_spectral_gap(adj, mode="out"),
        "shannon_entropy_entries": shannon_entropy_entries(adj),
        "degree_distribution_entropy_in": degree_distribution_entropy(adj, mode="in"),
        "degree_distribution_entropy_out": degree_distribution_entropy(adj, mode="out"),
        "random_walk_entropy": random_walk_entropy(adj),
        "von_neumann_entropy_sym": von_neumann_entropy(adj, mode="all"),
        "von_neumann_entropy_in": von_neumann_entropy(adj, mode="in"),
        "von_neumann_entropy_out": von_neumann_entropy(adj, mode="out"),
        "topological_entropy": topological_entropy(adj),
    }
    if advanced:
        out.update(_advanced_metrics(adj))
    return out
