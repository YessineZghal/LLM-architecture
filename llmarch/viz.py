import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import torch


def plot_losses(hist, title="Training curve"):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(hist["step"], hist["train"], alpha=.5, label="train")
    ax.plot(hist["val_step"], hist["val"], "o-", label="val")
    ax.set(xlabel="step", ylabel="cross-entropy", title=title); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.show()


def plot_heatmap(m, xlabels=None, ylabels=None, title="", cmap="viridis", ax=None, figsize=(5, 4)):
    m = m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(m, cmap=cmap, aspect="auto")
    if xlabels is not None:
        ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
    if ylabels is not None:
        ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_title(title); plt.colorbar(im, ax=ax, fraction=.046)
    return ax


PALETTE = {"embed": "#cfe8ff", "attn": "#ffd9a8", "ffn": "#d5f0c8", "norm": "#eeeeee",
           "moe": "#f6c6e8", "ssm": "#d9d0ff", "rnn": "#ffe9a0", "out": "#ffc9c9", "other": "#e0e0e0"}


def block_diagram(blocks, title="", repeat=None, figsize=(4.2, None), skip=None):
    """Vertical block diagram (bottom -> top). blocks: list of (label, kind) with kind in PALETTE.
    repeat: e.g. 'N x' draws a bracket around blocks flagged with kind suffix '*' (e.g. 'attn*')."""
    n = len(blocks)
    h = figsize[1] or 0.75 * n + 1
    fig, ax = plt.subplots(figsize=(figsize[0], h))
    ax.set_xlim(0, 10); ax.set_ylim(-0.5, n + 0.2); ax.axis("off")
    rep = []
    for i, (label, kind) in enumerate(blocks):
        star = kind.endswith("*"); k = kind.rstrip("*")
        if star: rep.append(i)
        ax.add_patch(FancyBboxPatch((1.5, i), 7, 0.62, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=PALETTE.get(k, "#ddd"), ec="#444", lw=1))
        ax.text(5, i + .31, label, ha="center", va="center", fontsize=9)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((5, i + .64), (5, i + 1), arrowstyle="-|>", mutation_scale=9, color="#444"))
    if rep and repeat:
        lo, hi = min(rep), max(rep)
        ax.add_patch(FancyBboxPatch((0.9, lo - .08), 8.2, hi - lo + .78, boxstyle="round,pad=0.02,rounding_size=0.2",
                                    fc="none", ec="#c0392b", ls="--", lw=1.3))
        ax.text(9.2, (lo + hi) / 2 + .3, repeat, rotation=90, va="center", color="#c0392b", fontsize=10)
    ax.set_title(title, fontsize=11)
    plt.tight_layout(); plt.show()


def show_summary(model, input_data=None, input_size=None, depth=3, **kw):
    """torchinfo summary (tables of layers, shapes, params)."""
    from torchinfo import summary
    kwargs = dict(depth=depth, col_names=("input_size", "output_size", "num_params"), verbose=0, **kw)
    if input_data is not None:
        s = summary(model, input_data=input_data, **kwargs)
    else:
        s = summary(model, input_size=input_size, **kwargs)
    print(s); return s


def export_onnx(model, example_inputs, path, **kw):
    """Export to ONNX so the graph can be opened with `uv run netron <path>`. Returns the path or None."""
    try:
        torch.onnx.export(model.eval(), example_inputs, str(path), dynamo=False, **kw)
        return path
    except Exception as e:  # tracing can fail for exotic ops - not fatal
        print("ONNX export skipped:", type(e).__name__, str(e)[:150]); return None


def show_source(*objs):
    """Pretty-print the source code of classes/functions inside a notebook."""
    import inspect
    from IPython.display import Markdown, display
    display(Markdown("\n".join(f"```python\n{inspect.getsource(o)}\n```" for o in objs)))


def fx_graph_plot(module, figsize=(6, 7), title="torch.fx computation graph"):
    """Trace with torch.fx and draw the dataflow graph with networkx (no Graphviz needed)."""
    import torch.fx as fx, networkx as nx
    gm = fx.symbolic_trace(module)
    G = nx.DiGraph(); labels = {}; colors = []
    for n in gm.graph.nodes:
        G.add_node(n.name); labels[n.name] = (str(n.target).split(".")[-1] if n.op != "placeholder" else n.name)[:18]
        for a in n.all_input_nodes:
            G.add_edge(a.name, n.name)
    cmap = {"placeholder": "#cfe8ff", "call_module": "#d5f0c8", "call_function": "#ffd9a8", "call_method": "#f6c6e8",
            "output": "#ffc9c9", "get_attr": "#eee"}
    colors = [cmap[n.op] for n in gm.graph.nodes]
    order = {n.name: i for i, n in enumerate(gm.graph.nodes)}
    layers = {}
    for node in nx.topological_sort(G):
        layers[node] = max([layers[p] + 1 for p in G.predecessors(node)], default=0)
    for node, l in layers.items():
        G.nodes[node]["layer"] = l
    pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
    pos = {k: (x, -y) for k, (x, y) in pos.items()}
    plt.figure(figsize=figsize)
    nx.draw(G, pos, labels=labels, node_color=colors, node_size=1500, font_size=7, edgecolors="#444", arrowsize=10)
    plt.title(title); plt.show(); return gm
