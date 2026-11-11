"""Compare post-training and quantization-aware training on MNIST datasets.

The app offers MNIST and Fashion-MNIST, using each dataset's complete
60,000-image training split and 10,000-image test split. Its model is a
small fully connected network (MLP): each 28x28 grayscale image is
flattened into 784 inputs, passed through a linear hidden layer with ReLU,
then mapped to 10 output scores. The hidden-layer width is configurable.

The experiment compares weights at bit-widths from 32 bits down to 2 bits.
Post-training quantization (PTQ) first trains a 32-bit baseline, then
evaluates copies of that model with quantized weights. Quantization-aware
training (QAT) trains each lower-bit model with quantization simulated
during its forward pass; it can start from the baseline or from a new
initialization. Results are reported across the selected random seeds.

Fake quantization rounds and clamps weights to symmetric, uniformly spaced
levels, then scales them back to floating-point values. During QAT,
backpropagation uses a straight-through estimator (STE), treating the
rounding operation as identity for gradients. Only weights are quantized;
biases and activations remain FP32. This models the effect of quantization
on predictions, but does not store integer weights or provide the runtime
and memory benefits of an integer-quantized deployment.
"""

import io
import time

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "./data"
BIT_WIDTHS = [32, 16, 8, 7, 6, 5, 4, 3, 2]
DATASETS = {
    "MNIST": (datasets.MNIST, 0.1307, 0.3081),
    "Fashion-MNIST": (datasets.FashionMNIST, 0.2860, 0.3530),
}

_data_cache = {}


def get_data(dataset_name):
    """Load a selected dataset once into device tensors."""
    if dataset_name not in _data_cache:
        dataset_class, mean, std = DATASETS[dataset_name]
        tr = dataset_class(DATA_ROOT, train=True, download=True)
        te = dataset_class(DATA_ROOT, train=False, download=True)

        def prep(ds):
            x = (ds.data.float() / 255.0 - mean) / std
            return x.unsqueeze(1).to(DEVICE), ds.targets.to(DEVICE)

        _data_cache[dataset_name] = (prep(tr), prep(te))
    return _data_cache[dataset_name]


class FakeQuantizeSTE(torch.autograd.Function):
    """Symmetric uniform fake quantization with a straight-through gradient."""

    @staticmethod
    def forward(ctx, tensor, bits, per_channel):
        if bits >= 32:
            return tensor
        qmax = 2 ** (bits - 1) - 1
        if per_channel:
            max_val = tensor.abs().amax(dim=1, keepdim=True)
        else:
            max_val = tensor.abs().max()
        scale = (max_val / qmax).clamp_min(1e-12)
        return torch.clamp(torch.round(tensor / scale), -qmax, qmax) * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class QuantizedFCN(nn.Module):
    def __init__(self, bits=32, per_channel=False, hidden=24):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.bits = bits
        self.per_channel = per_channel

    def forward(self, x):
        x = x.view(x.size(0), -1)
        w1 = FakeQuantizeSTE.apply(self.fc1.weight, self.bits, self.per_channel)
        w2 = FakeQuantizeSTE.apply(self.fc2.weight, self.bits, self.per_channel)
        x = F.relu(F.linear(x, w1, self.fc1.bias))
        return F.linear(x, w2, self.fc2.bias)


def train_model(model, epochs, lr, dataset_name, batch_size=64):
    (xtr, ytr), _ = get_data(dataset_name)
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate_model(model, dataset_name):
    _, (xte, yte) = get_data(dataset_name)
    model.eval()
    correct = 0
    for i in range(0, len(xte), 2000):
        correct += (model(xte[i:i + 2000]).argmax(1) == yte[i:i + 2000]).sum().item()
    return 100.0 * correct / len(xte)


def make_plots(dataset_name, bits, ptq, qat_mean, qat_std):
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(bits))
    labels = [str(b) for b in bits]

    ax.plot(xs, ptq, marker="o", linestyle="--", color="crimson", label="PTQ")
    ax.errorbar(xs, qat_mean, yerr=qat_std, marker="s", color="teal", capsize=3,
                label="QAT (mean, std over seeds)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Weight bit-depth")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(f"{dataset_name}: PTQ vs QAT accuracy")
    ax.set_ylim(80, 100)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    fig.tight_layout()
    return fig


def run_experiment(dataset_name, epochs, lr, n_seeds, per_channel, qat_init, hidden,
                   progress=gr.Progress()):
    epochs, n_seeds, hidden = int(epochs), int(n_seeds), int(hidden)
    per_channel = per_channel == "Per-channel"
    log = []
    t0 = time.time()

    # PTQ: the baseline is trained once per seed, then quantized at each bit-depth
    ptq_runs = {b: [] for b in BIT_WIDTHS}
    qat_runs = {b: [] for b in BIT_WIDTHS}
    total = n_seeds * (1 + len(BIT_WIDTHS) - 1)
    done = 0

    for s in range(n_seeds):
        torch.manual_seed(s)
        base = QuantizedFCN(32, per_channel, hidden).to(DEVICE)
        train_model(base, epochs, lr, dataset_name)
        state = {k: v.clone() for k, v in base.state_dict().items()}
        fp32_acc = evaluate_model(base, dataset_name)
        done += 1
        progress(done / total, desc=f"Seed {s}: baseline trained")

        for b in BIT_WIDTHS:
            m = QuantizedFCN(b, per_channel, hidden).to(DEVICE)
            m.load_state_dict(state)
            ptq_runs[b].append(evaluate_model(m, dataset_name))

        for b in BIT_WIDTHS:
            if b == 32:
                qat_runs[b].append(fp32_acc)
                continue
            torch.manual_seed(1000 + s)
            m = QuantizedFCN(b, per_channel, hidden).to(DEVICE)
            if qat_init == "Fine-tune from FP32 baseline":
                m.load_state_dict(state)
            train_model(m, epochs, lr, dataset_name)
            qat_runs[b].append(evaluate_model(m, dataset_name))
            done += 1
            progress(done / total, desc=f"Seed {s}: QAT {b}-bit")

    ptq = [float(np.mean(ptq_runs[b])) for b in BIT_WIDTHS]
    qat_mean = [float(np.mean(qat_runs[b])) for b in BIT_WIDTHS]
    qat_std = [float(np.std(qat_runs[b])) for b in BIT_WIDTHS]

    rows = [[b, round(ptq[i], 2), round(qat_mean[i], 2), round(qat_std[i], 2)]
            for i, b in enumerate(BIT_WIDTHS)]
    log.append(f"Finished in {time.time() - t0:.1f} s on {DEVICE}.")
    fig = make_plots(dataset_name, BIT_WIDTHS, ptq, qat_mean, qat_std)
    return fig, rows, "\n".join(log)


with gr.Blocks(title="MNIST QAT vs PTQ") as demo:
    gr.Markdown("# MNIST datasets: Quantization-Aware Training vs Post-Training Quantization\n"
                "Weights only are fake-quantized (symmetric, uniform). Biases and activations stay in FP32.")
    with gr.Row():
        dataset = gr.Dropdown(list(DATASETS), value="MNIST", label="Dataset")
        epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
        lr = gr.Number(value=0.002, label="Adam learning rate")
        n_seeds = gr.Slider(1, 5, value=1, step=1, label="Seeds (averaged)")
        hidden = gr.Slider(8, 128, value=24, step=8, label="Hidden units")
    with gr.Row():
        granularity = gr.Radio(["Per-tensor", "Per-channel"], value="Per-channel", label="Scale granularity")
        qat_init = gr.Radio(["Train from scratch", "Fine-tune from FP32 baseline"],
                            value="Fine-tune from FP32 baseline", label="QAT initialisation")
    btn = gr.Button("Run experiment", variant="primary")
    plot = gr.Plot(label="Results")
    table = gr.Dataframe(headers=["Bits", "PTQ acc (%)", "QAT mean acc (%)", "QAT std"], label="Results table")
    status = gr.Textbox(label="Status", interactive=False)
    btn.click(run_experiment, [dataset, epochs, lr, n_seeds, granularity, qat_init, hidden],
              [plot, table, status])

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
