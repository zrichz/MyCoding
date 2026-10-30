# Vision transformer classifier on MNIST, testing Muon+AdamW across patch sizes
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

seed = 1234
batch_size = 512
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Patch sizes to test (each must divide 28 exactly)
#PATCH_SIZES = (2, 4, 7, 14)
PATCH_SIZES = (4,)

# Optimisers to compare, and their learning rates
OPTIMIZERS = {
    "Muon+AdamW": 1e-3,
}

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2000, shuffle=False)


class ViT(nn.Module):
    """Images are split into patches (tokens), a class token is prepended, and a
    transformer encoder feeds the class token into a linear classification head."""

    def __init__(self, image_size=28, patch_size=4, dim=64, heads=4, depth=3,
                 ff_mult=4, num_classes=10):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        n_tokens = self.grid ** 2

        self.patch_embed = nn.Linear(patch_size ** 2, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * ff_mult,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def patchify(self, x):
        b, p, g = x.shape[0], self.patch_size, self.grid
        x = x.view(b, g, p, g, p).permute(0, 1, 3, 2, 4)
        return x.reshape(b, g * g, p * p)

    def forward(self, images):
        x = self.patch_embed(self.patchify(images.view(-1, self.image_size, self.image_size)))
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        x = self.norm(self.encoder(x)[:, 0])
        return self.head(x)


def make_optimizer(name, model, lr):
    if name == "Muon+AdamW":
        # Muon only handles 2D hidden weight matrices (attention and feed-forward
        # inside the encoder). Embeddings, class token, norms, biases and the head
        # go to AdamW. Muon's lr is rescaled to match AdamW's update RMS so the
        # same lr can be used for both.
        muon_params, adam_params = [], []
        for pname, p in model.named_parameters():
            if p.ndim == 2 and pname.startswith("encoder."):
                muon_params.append(p)
            else:
                adam_params.append(p)
        return [
            torch.optim.Muon(muon_params, lr=lr, weight_decay=0.01,
                             adjust_lr_fn="match_rms_adamw"),
            torch.optim.AdamW(adam_params, lr=lr),
        ]
    raise ValueError(name)


def evaluate(model, lossfn):
    """Loss and classification accuracy (%) over the full 10000-image test set."""
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            total_loss += lossfn(logits, labels).item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
    model.train()
    n = len(test_dataset)
    return total_loss / n, 100 * correct / n


def train(patch_size, name, lr, index, total_runs):
    torch.manual_seed(seed)  # identical init and data order for every optimiser
    model = ViT(patch_size=patch_size).to(device)
    optimizers = make_optimizer(name, model, lr)
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    lossfn = nn.CrossEntropyLoss()

    batch_losses, train_accs, test_losses, test_accs, epoch_times = [], [], [], [], []
    train_time = 0.0
    for epoch in range(epochs):
        epoch_start = time.time()
        correct, seen = 0, 0
        bar = tqdm(loader, leave=False,
                   desc=f"[{index}/{total_runs}] patch {patch_size} {name} epoch {epoch + 1}/{epochs}")
        for data, labels in bar:
            data, labels = data.to(device), labels.to(device)
            logits = model(data)
            loss = lossfn(logits, labels)
            loss.backward()
            for opt in optimizers:
                opt.step()
                opt.zero_grad()
            batch_losses.append(loss.item())
            correct += (logits.argmax(1) == labels).sum().item()
            seen += len(labels)
            bar.set_postfix(loss=f"{loss.item():.4f}")
        train_time += time.time() - epoch_start
        train_accs.append(100 * correct / seen)

        test_loss, test_acc = evaluate(model, lossfn)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        epoch_times.append(train_time)
        tqdm.write(f"[{index}/{total_runs}] patch {patch_size} {name:12s} epoch {epoch + 1}/{epochs}, "
                   f"train acc = {train_accs[-1]:.2f}%, test loss = {test_loss:.4f}, "
                   f"test acc = {test_acc:.2f}%, train time {train_time:.0f}s")

    return {"batch_losses": batch_losses, "train_accs": train_accs,
            "test_losses": test_losses, "test_accs": test_accs,
            "times": epoch_times, "train_time": train_time}


runs = [(ps, name) for ps in PATCH_SIZES for name in OPTIMIZERS]
results = {}
for i, (ps, name) in enumerate(runs, start=1):
    results[(ps, name)] = train(ps, name, OPTIMIZERS[name], i, len(runs))

print("\nPatch  Tokens  Optimiser     lr       train acc %   test acc %   best test acc %   test loss   time (s)")
for (ps, name), r in results.items():
    print(f"{ps:<5d}  {(28 // ps) ** 2:<6d}  {name:12s}  {OPTIMIZERS[name]:<7g}  "
          f"{r['train_accs'][-1]:<12.2f}  {r['test_accs'][-1]:<11.2f}  "
          f"{max(r['test_accs']):<15.2f}   {r['test_losses'][-1]:<9.4f}   {r['train_time']:.1f}")

# Charts
EMA_PERIOD = 20


def ema(values, period):
    alpha = 2 / (period + 1)
    out, current = [], values[0]
    for v in values:
        current = alpha * v + (1 - alpha) * current
        out.append(current)
    return out


fig, axes = plt.subplots(2, 3, figsize=(21, 10))
ax_loss, ax_test_loss, ax_train_acc, ax_acc, ax_acc_time, ax_blank = axes.flatten()
epoch_axis = range(1, epochs + 1)
# Colour identifies the patch size, line style identifies the optimiser
patch_colors = {ps: f"C{k}" for k, ps in enumerate(PATCH_SIZES)}
opt_styles = dict(zip(OPTIMIZERS, ("-", "--", ":", "-.")))
for (ps, name), r in results.items():
    style = dict(color=patch_colors[ps], linestyle=opt_styles[name])
    label = f"patch {ps} {name}"
    ax_loss.plot(r["batch_losses"], alpha=0.15, color=style["color"])
    ax_loss.plot(ema(r["batch_losses"], EMA_PERIOD), linewidth=2,
                 label=f"{label} (EMA {EMA_PERIOD})", **style)
    ax_test_loss.plot(epoch_axis, r["test_losses"], marker="o", label=label, **style)
    ax_train_acc.plot(epoch_axis, r["train_accs"], marker="o", label=label, **style)
    ax_acc.plot(epoch_axis, r["test_accs"], marker="o", label=label, **style)
    ax_acc_time.plot(r["times"], r["test_accs"], marker="o", label=label, **style)
for name in OPTIMIZERS:
    ax_blank.plot(PATCH_SIZES, [results[(ps, name)]["test_accs"][-1] for ps in PATCH_SIZES],
                  marker="o", linestyle=opt_styles[name], color="black", label=name)
ax_blank.set_title("Final Test Accuracy vs patch size")
ax_blank.set_xlabel("patch size")
ax_blank.set_xticks(PATCH_SIZES)
ax_blank.set_ylabel("Accuracy (%)")
ax_blank.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

ax_loss.set_title("Training Loss")
ax_loss.set_xlabel("step")
ax_loss.set_ylabel("Cross-entropy")
ax_loss.set_yscale("log")
ax_test_loss.set_title("Test Loss")
ax_test_loss.set_xlabel("epoch")
ax_test_loss.set_ylabel("Cross-entropy")
ax_train_acc.set_title("Training Accuracy")
ax_train_acc.set_xlabel("epoch")
ax_acc.set_title("Test Accuracy")
ax_acc.set_xlabel("epoch")
ax_acc_time.set_title("Test Accuracy vs training wall time")
ax_acc_time.set_xlabel("training wall time (s)")
for ax in (ax_train_acc, ax_acc, ax_acc_time):
    ax.set_ylabel("Accuracy (%)")
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
for ax in axes.flatten():
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
