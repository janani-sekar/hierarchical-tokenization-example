"""Train and compare flat vs HTP-style hierarchical models on arithmetic."""

import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data import generate_addition_dataset
from tokenizer import (
    decode, VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID,
    build_flat_sequence, build_hierarchical_sequence,
)
from model import ArithmeticTransformer


# -- Datasets ----------------------------------------------------------------

class FlatDataset(Dataset):
    """Character-level sequences for next-token prediction."""

    def __init__(self, data: list, max_seq_len: int):
        self.samples = []
        for expr, answer in data:
            tokens, prefix_len = build_flat_sequence(expr, answer)
            self.samples.append((tokens, prefix_len))
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, prefix_len = self.samples[idx]
        n = len(tokens)

        padded = tokens + [PAD_ID] * (self.max_seq_len - n)
        pad_mask = [False] * n + [True] * (self.max_seq_len - n)

        # Next-token prediction: input = all but last, target = all but first
        inp = padded[:-1]
        tgt = padded[1:]
        pm = pad_mask[:-1]

        # Loss mask: only on answer tokens + EOS (positions prefix_len-1 .. n-2 in target)
        loss_mask = [False] * (self.max_seq_len - 1)
        for j in range(prefix_len - 1, n - 1):
            loss_mask[j] = True

        return {
            'input': torch.tensor(inp, dtype=torch.long),
            'target': torch.tensor(tgt, dtype=torch.long),
            'pad_mask': torch.tensor(pm, dtype=torch.bool),
            'loss_mask': torch.tensor(loss_mask, dtype=torch.bool),
            'prefix_len': prefix_len,
        }


class HierarchicalDataset(Dataset):
    """Sequences with <num> tokens and rewire info for HTP."""

    def __init__(self, data: list, max_seq_len: int, max_rewires: int = 2):
        self.samples = []
        for expr, answer in data:
            tokens, prefix_len, rewire_pairs = build_hierarchical_sequence(expr, answer)
            self.samples.append((tokens, prefix_len, rewire_pairs))
        self.max_seq_len = max_seq_len
        self.max_rewires = max_rewires

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, prefix_len, rewire_pairs = self.samples[idx]
        n = len(tokens)

        padded = tokens + [PAD_ID] * (self.max_seq_len - n)
        pad_mask = [False] * n + [True] * (self.max_seq_len - n)

        inp = padded[:-1]
        tgt = padded[1:]
        pm = pad_mask[:-1]

        loss_mask = [False] * (self.max_seq_len - 1)
        for j in range(prefix_len - 1, n - 1):
            loss_mask[j] = True

        # Pad rewire pairs to fixed size (pad with (0,0) which is a no-op: copy BOS to BOS)
        dst = [p[0] for p in rewire_pairs]
        src = [p[1] for p in rewire_pairs]
        while len(dst) < self.max_rewires:
            dst.append(0)
            src.append(0)

        return {
            'input': torch.tensor(inp, dtype=torch.long),
            'target': torch.tensor(tgt, dtype=torch.long),
            'pad_mask': torch.tensor(pm, dtype=torch.bool),
            'loss_mask': torch.tensor(loss_mask, dtype=torch.bool),
            'prefix_len': prefix_len,
            'rewire_src': torch.tensor(src, dtype=torch.long),
            'rewire_dst': torch.tensor(dst, dtype=torch.long),
        }


# -- Training ----------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, criterion, device, hierarchical=False):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        inp = batch['input'].to(device)
        tgt = batch['target'].to(device)
        pad_mask = batch['pad_mask'].to(device)
        loss_mask = batch['loss_mask'].to(device)

        kwargs = {}
        if hierarchical:
            kwargs['rewire_src'] = batch['rewire_src'].to(device)
            kwargs['rewire_dst'] = batch['rewire_dst'].to(device)

        logits = model(inp, padding_mask=pad_mask, **kwargs)

        # Loss only on answer positions
        active = loss_mask.reshape(-1)
        loss = criterion(logits.reshape(-1, logits.size(-1))[active], tgt.reshape(-1)[active])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# -- Evaluation (teacher-forced, fast) ---------------------------------------

@torch.no_grad()
def evaluate_tf(model, dataloader, device, hierarchical=False):
    """Teacher-forced exact-match accuracy (fast, batched)."""
    model.eval()
    correct = total = 0

    for batch in dataloader:
        inp = batch['input'].to(device)
        tgt = batch['target'].to(device)
        pad_mask = batch['pad_mask'].to(device)
        loss_mask = batch['loss_mask']

        kwargs = {}
        if hierarchical:
            kwargs['rewire_src'] = batch['rewire_src'].to(device)
            kwargs['rewire_dst'] = batch['rewire_dst'].to(device)

        logits = model(inp, padding_mask=pad_mask, **kwargs)
        preds = logits.argmax(dim=-1).cpu()

        for i in range(inp.size(0)):
            mask = loss_mask[i]
            if mask.any() and (preds[i][mask] == batch['target'][i][mask]).all():
                correct += 1
            total += 1

    return correct / total


# -- Evaluation (autoregressive, accurate) -----------------------------------

@torch.no_grad()
def evaluate_ar(model, dataset, device, hierarchical=False, max_gen=5):
    """Autoregressive exact-match accuracy (one example at a time)."""
    model.eval()
    correct = total = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        prefix_len = sample['prefix_len']

        tokens = sample['input'][:prefix_len].unsqueeze(0).to(device)

        kwargs = {}
        if hierarchical:
            kwargs['rewire_src'] = sample['rewire_src'].unsqueeze(0).to(device)
            kwargs['rewire_dst'] = sample['rewire_dst'].unsqueeze(0).to(device)

        generated = tokens
        for _ in range(max_gen):
            logits = model(generated, **kwargs)
            next_tok = logits[0, -1].argmax().item()
            if next_tok == EOS_ID:
                break
            generated = torch.cat(
                [generated, torch.tensor([[next_tok]], device=device)], dim=1,
            )

        pred_ids = generated[0, prefix_len:].tolist()

        # True answer from target (answer tokens before EOS)
        true_ids = []
        for j in range(len(sample['target'])):
            if sample['loss_mask'][j]:
                tok = sample['target'][j].item()
                if tok == EOS_ID:
                    break
                true_ids.append(tok)

        if pred_ids == true_ids:
            correct += 1
        total += 1

    return correct / total


def show_examples(model, dataset, device, hierarchical=False, n=8, max_gen=5):
    """Print a few example predictions."""
    model.eval()

    for i in range(min(n, len(dataset))):
        sample = dataset[i]
        prefix_len = sample['prefix_len']
        tokens = sample['input'][:prefix_len].unsqueeze(0).to(device)

        kwargs = {}
        if hierarchical:
            kwargs['rewire_src'] = sample['rewire_src'].unsqueeze(0).to(device)
            kwargs['rewire_dst'] = sample['rewire_dst'].unsqueeze(0).to(device)

        generated = tokens
        for _ in range(max_gen):
            logits = model(generated, **kwargs)
            next_tok = logits[0, -1].argmax().item()
            if next_tok == EOS_ID:
                break
            generated = torch.cat(
                [generated, torch.tensor([[next_tok]], device=device)], dim=1,
            )

        pred_str = decode(generated[0, prefix_len:].tolist())

        true_ids = []
        for j in range(len(sample['target'])):
            if sample['loss_mask'][j]:
                tok = sample['target'][j].item()
                if tok == EOS_ID:
                    break
                true_ids.append(tok)
        true_str = decode(true_ids)

        expr_str = decode(sample['input'][:prefix_len].tolist())
        marker = "ok" if pred_str == true_str else "WRONG"
        print(f"  {expr_str} {pred_str}  (expected {true_str}) [{marker}]")


# -- Plotting ----------------------------------------------------------------

def save_plots(history: dict, plot_dir: str):
    """Save training loss and accuracy curves."""
    epochs = history['epoch']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss
    ax1.plot(epochs, history['flat_loss'], label='Flat', linewidth=1.5)
    ax1.plot(epochs, history['hier_loss'], label='Hierarchical', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['flat_acc'], label='Flat', linewidth=1.5)
    ax2.plot(epochs, history['hier_acc'], label='Hierarchical', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Exact-match Accuracy')
    ax2.set_title('Test Accuracy (teacher-forced)')
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(plot_dir, 'training_curves.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {path}")


# -- Main --------------------------------------------------------------------

def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f"Device: {device}")

    train_data = generate_addition_dataset(50_000, min_val=1, max_val=999, seed=0)
    test_data = generate_addition_dataset(5_000, min_val=1, max_val=999, seed=1)

    d_model = 64
    nhead = 4
    num_layers = 4
    d_ff = 256
    batch_size = 128
    lr = 3e-4
    epochs = 50

    # Max sequence lengths (with padding buffer)
    # Flat longest:  "999+999=1998" -> [BOS,9,9,9,+,9,9,9,=,1,9,9,8,EOS] = 14
    # Hier longest:  same -> [BOS,<num>,9,9,9,+,<num>,9,9,9,=,1,9,9,8,EOS] = 16
    flat_max = 15
    hier_max = 17
    model_max = 24  # enough for positional encoding during generation

    flat_train_ds = FlatDataset(train_data, max_seq_len=flat_max)
    flat_test_ds = FlatDataset(test_data, max_seq_len=flat_max)
    hier_train_ds = HierarchicalDataset(train_data, max_seq_len=hier_max)
    hier_test_ds = HierarchicalDataset(test_data, max_seq_len=hier_max)

    flat_train_dl = DataLoader(flat_train_ds, batch_size=batch_size, shuffle=True)
    flat_test_dl = DataLoader(flat_test_ds, batch_size=batch_size)
    hier_train_dl = DataLoader(hier_train_ds, batch_size=batch_size, shuffle=True)
    hier_test_dl = DataLoader(hier_test_ds, batch_size=batch_size)

    # Identical architecture, identical parameter count
    flat_model = ArithmeticTransformer(
        VOCAB_SIZE, d_model=d_model, nhead=nhead, num_layers=num_layers,
        d_ff=d_ff, max_seq_len=model_max,
    ).to(device)

    hier_model = ArithmeticTransformer(
        VOCAB_SIZE, d_model=d_model, nhead=nhead, num_layers=num_layers,
        d_ff=d_ff, max_seq_len=model_max,
    ).to(device)

    n_params = sum(p.numel() for p in flat_model.parameters())
    print(f"Parameters per model: {n_params:,}")
    print()

    flat_opt = torch.optim.Adam(flat_model.parameters(), lr=lr)
    hier_opt = torch.optim.Adam(hier_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Tracking
    history = {
        'epoch': [],
        'flat_loss': [], 'hier_loss': [],
        'flat_acc': [], 'hier_acc': [],
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        flat_loss = train_epoch(
            flat_model, flat_train_dl, flat_opt, criterion, device, hierarchical=False,
        )
        hier_loss = train_epoch(
            hier_model, hier_train_dl, hier_opt, criterion, device, hierarchical=True,
        )

        flat_acc = evaluate_tf(flat_model, flat_test_dl, device, hierarchical=False)
        hier_acc = evaluate_tf(hier_model, hier_test_dl, device, hierarchical=True)

        elapsed = time.time() - t0

        history['epoch'].append(epoch)
        history['flat_loss'].append(flat_loss)
        history['hier_loss'].append(hier_loss)
        history['flat_acc'].append(flat_acc)
        history['hier_acc'].append(hier_acc)

        print(
            f"Epoch {epoch:3d} | "
            f"Flat: loss={flat_loss:.4f} acc={flat_acc:.4f} | "
            f"Hier: loss={hier_loss:.4f} acc={hier_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

    # Final autoregressive evaluation
    print("\nFinal results (autoregressive):")
    flat_acc_ar = evaluate_ar(flat_model, flat_test_ds, device, hierarchical=False, max_gen=5)
    hier_acc_ar = evaluate_ar(hier_model, hier_test_ds, device, hierarchical=True, max_gen=5)
    print(f"  Flat exact-match accuracy:         {flat_acc_ar:.4f}")
    print(f"  Hierarchical exact-match accuracy:  {hier_acc_ar:.4f}")

    print("\nFlat model examples:")
    show_examples(flat_model, flat_test_ds, device, hierarchical=False)

    print("\nHierarchical model examples:")
    show_examples(hier_model, hier_test_ds, device, hierarchical=True)

    # Save plots
    plot_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    save_plots(history, plot_dir)


if __name__ == '__main__':
    main()
