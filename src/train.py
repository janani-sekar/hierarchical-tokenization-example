"""Train and compare six model variants on arithmetic.

Grid: digit ordering (L2R, R2L) x attention structure (flat, HTP, HTP+HAttn).
Each variant trains independently with early stopping on validation accuracy.
"""

import argparse
import csv
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Disable SDPA backends that cause CUBLAS errors on older GPUs (e.g. T4)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from data import generate_addition_dataset, generate_mixed_dataset, generate_algebraic_dataset
from tokenizer import (
    decode, reverse_digits, VOCAB_SIZE, PAD_ID, BOS_ID, EOS_ID,
    build_flat_sequence, build_r2l_flat_sequence,
    build_hierarchical_sequence, build_r2l_hier_sequence,
)
from model import ArithmeticTransformer, HierAttnTransformer


# -- Datasets ----------------------------------------------------------------

class FlatDataset(Dataset):
    """Character-level sequences for next-token prediction."""

    def __init__(self, data, max_seq_len, build_fn=build_flat_sequence):
        self.samples = []
        for expr, answer in data:
            tokens, prefix_len = build_fn(expr, answer)
            self.samples.append((tokens, prefix_len))
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, prefix_len = self.samples[idx]
        n = len(tokens)

        padded = tokens + [PAD_ID] * (self.max_seq_len - n)
        pad_mask = [False] * n + [True] * (self.max_seq_len - n)

        inp = padded[:-1]
        tgt = padded[1:]
        pm = pad_mask[:-1]

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


class HierDataset(Dataset):
    """Sequences with <num> tokens, rewire info, and group IDs."""

    def __init__(self, data, max_seq_len, build_fn=build_hierarchical_sequence, max_rewires=2):
        self.samples = []
        for expr, answer in data:
            tokens, prefix_len, rewire_pairs, group_ids = build_fn(expr, answer)
            self.samples.append((tokens, prefix_len, rewire_pairs, group_ids))
        self.max_seq_len = max_seq_len
        self.max_rewires = max_rewires

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, prefix_len, rewire_pairs, group_ids = self.samples[idx]
        n = len(tokens)

        padded = tokens + [PAD_ID] * (self.max_seq_len - n)
        pad_mask = [False] * n + [True] * (self.max_seq_len - n)

        inp = padded[:-1]
        tgt = padded[1:]
        pm = pad_mask[:-1]

        loss_mask = [False] * (self.max_seq_len - 1)
        for j in range(prefix_len - 1, n - 1):
            loss_mask[j] = True

        # Pad rewire pairs (pad with (0,0) = copy BOS to BOS = no-op)
        dst = [p[0] for p in rewire_pairs]
        src = [p[1] for p in rewire_pairs]
        while len(dst) < self.max_rewires:
            dst.append(0)
            src.append(0)

        # Pad group_ids and slice to match input (all but last token)
        padded_gids = group_ids + [-1] * (self.max_seq_len - len(group_ids))
        inp_gids = padded_gids[:-1]

        return {
            'input': torch.tensor(inp, dtype=torch.long),
            'target': torch.tensor(tgt, dtype=torch.long),
            'pad_mask': torch.tensor(pm, dtype=torch.bool),
            'loss_mask': torch.tensor(loss_mask, dtype=torch.bool),
            'prefix_len': prefix_len,
            'rewire_src': torch.tensor(src, dtype=torch.long),
            'rewire_dst': torch.tensor(dst, dtype=torch.long),
            'group_ids': torch.tensor(inp_gids, dtype=torch.long),
        }


# -- Training ----------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, criterion, device,
                is_hier=False, is_hattn=False, desc=''):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=desc, leave=False):
        optimizer.zero_grad()

        inp = batch['input'].to(device)
        tgt = batch['target'].to(device)
        pad_mask = batch['pad_mask'].to(device)
        loss_mask = batch['loss_mask'].to(device)

        kwargs = {}
        if is_hier:
            kwargs['rewire_src'] = batch['rewire_src'].to(device)
            kwargs['rewire_dst'] = batch['rewire_dst'].to(device)
        if is_hattn:
            kwargs['group_ids'] = batch['group_ids'].to(device)

        logits = model(inp, padding_mask=pad_mask, **kwargs)

        active = loss_mask.reshape(-1)
        loss = criterion(logits.reshape(-1, logits.size(-1))[active], tgt.reshape(-1)[active])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# -- Evaluation (teacher-forced, fast) ---------------------------------------

@torch.no_grad()
def evaluate_tf(model, dataloader, device, is_hier=False, is_hattn=False):
    """Teacher-forced exact-match accuracy (fast, batched)."""
    model.eval()
    correct = total = 0

    for batch in dataloader:
        inp = batch['input'].to(device)
        pad_mask = batch['pad_mask'].to(device)
        loss_mask = batch['loss_mask']

        kwargs = {}
        if is_hier:
            kwargs['rewire_src'] = batch['rewire_src'].to(device)
            kwargs['rewire_dst'] = batch['rewire_dst'].to(device)
        if is_hattn:
            kwargs['group_ids'] = batch['group_ids'].to(device)

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
def evaluate_ar(model, dataset, device, is_hier=False, is_hattn=False,
                max_gen=5, desc=''):
    """Autoregressive exact-match accuracy (one example at a time)."""
    model.eval()
    correct = total = 0

    rng = tqdm(range(len(dataset)), desc=desc, leave=False) if desc else range(len(dataset))
    for i in rng:
        sample = dataset[i]
        prefix_len = sample['prefix_len']

        generated = sample['input'][:prefix_len].unsqueeze(0).to(device)

        kwargs = {}
        if is_hier:
            kwargs['rewire_src'] = sample['rewire_src'].unsqueeze(0).to(device)
            kwargs['rewire_dst'] = sample['rewire_dst'].unsqueeze(0).to(device)

        group_ids = None
        if is_hattn:
            group_ids = sample['group_ids'][:prefix_len].unsqueeze(0).to(device)

        for _ in range(max_gen):
            fwd_kwargs = dict(kwargs)
            if is_hattn:
                fwd_kwargs['group_ids'] = group_ids
            logits = model(generated, **fwd_kwargs)
            next_tok = logits[0, -1].argmax().item()
            if next_tok == EOS_ID:
                break
            generated = torch.cat(
                [generated, torch.tensor([[next_tok]], device=device)], dim=1,
            )
            if is_hattn:
                group_ids = torch.cat(
                    [group_ids, torch.tensor([[-1]], device=device)], dim=1,
                )

        pred_ids = generated[0, prefix_len:].tolist()

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


def show_examples(model, dataset, device, is_hier=False, is_hattn=False,
                  is_r2l=False, n=8, max_gen=5):
    """Print a few example predictions."""
    model.eval()

    for i in range(min(n, len(dataset))):
        sample = dataset[i]
        prefix_len = sample['prefix_len']
        generated = sample['input'][:prefix_len].unsqueeze(0).to(device)

        kwargs = {}
        if is_hier:
            kwargs['rewire_src'] = sample['rewire_src'].unsqueeze(0).to(device)
            kwargs['rewire_dst'] = sample['rewire_dst'].unsqueeze(0).to(device)

        group_ids = None
        if is_hattn:
            group_ids = sample['group_ids'][:prefix_len].unsqueeze(0).to(device)

        for _ in range(max_gen):
            fwd_kwargs = dict(kwargs)
            if is_hattn:
                fwd_kwargs['group_ids'] = group_ids
            logits = model(generated, **fwd_kwargs)
            next_tok = logits[0, -1].argmax().item()
            if next_tok == EOS_ID:
                break
            generated = torch.cat(
                [generated, torch.tensor([[next_tok]], device=device)], dim=1,
            )
            if is_hattn:
                group_ids = torch.cat(
                    [group_ids, torch.tensor([[-1]], device=device)], dim=1,
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

        # For R2L, reverse digits back to L2R for readable display
        if is_r2l:
            pred_str = reverse_digits(pred_str)
            true_str = reverse_digits(true_str)
            expr_str = reverse_digits(expr_str)

        marker = "ok" if pred_str == true_str else "WRONG"
        print(f"  {expr_str} {pred_str}  (expected {true_str}) [{marker}]")


# -- Plotting ----------------------------------------------------------------

# Color = attention structure, linestyle = digit ordering
VARIANT_STYLE = {
    'L2R':              {'color': '#1f77b4', 'ls': '-'},
    'L2R+HTP':          {'color': '#2ca02c', 'ls': '-'},
    'L2R+HTP+HAttn':    {'color': '#d62728', 'ls': '-'},
    'R2L':              {'color': '#1f77b4', 'ls': '--'},
    'R2L+HTP':          {'color': '#2ca02c', 'ls': '--'},
    'R2L+HTP+HAttn':    {'color': '#d62728', 'ls': '--'},
}


def save_plots(history, plot_dir, dgp_name=''):
    """Save training loss and accuracy curves for all variants."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, data in history.items():
        if not data['epoch']:
            continue
        style = VARIANT_STYLE[name]
        ax1.plot(data['epoch'], data['loss'], label=name,
                 color=style['color'], linestyle=style['ls'], linewidth=1.5)
        ax2.plot(data['epoch'], data['acc'], label=name,
                 color=style['color'], linestyle=style['ls'], linewidth=1.5)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Exact-match Accuracy')
    ax2.set_title('Validation Accuracy (teacher-forced)')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    suffix = f'_{dgp_name}' if dgp_name else ''
    path = os.path.join(plot_dir, f'training_curves{suffix}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {path}")


# -- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train 6 model variants on arithmetic tasks.")
    parser.add_argument("--dgp", choices=["addition", "mixed", "algebraic"], default="addition",
                        help="Data generating process (default: addition)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 3e-4; ignored if resuming)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Max epochs (default: 50 addition, 300 mixed/algebraic)")
    args = parser.parse_args()

    torch.manual_seed(2)

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f"Device: {device}")

    dgp = args.dgp

    # Data generation config per DGP
    DGP_DATA = {
        'addition': {
            'gen_fn': generate_addition_dataset,
            'key': 'min_val', 'max_key': 'max_val',
            'full_max': 999, 'ops': None,
            'flat_max': 17, 'hier_max': 19, 'max_gen': 5,
            'n_train': 50_000, 'n_eval': 5_000,
            'curriculum': None,  # addition already works well
        },
        'mixed': {
            'gen_fn': generate_mixed_dataset,
            'key': 'min_val', 'max_key': 'max_val',
            'full_max': 99, 'ops': "+-*/",
            'flat_max': 15, 'hier_max': 17, 'max_gen': 7,
            'n_train': 75_000, 'n_eval': 5_000,
            'curriculum': [25, 50, 99],
        },
        'algebraic': {
            'gen_fn': generate_algebraic_dataset,
            'key': 'min_coeff', 'max_key': 'max_coeff',
            'full_max': 99, 'ops': "+-*/",
            'flat_max': 19, 'hier_max': 21, 'max_gen': 8,
            'n_train': 75_000, 'n_eval': 5_000,
            'curriculum': [25, 50, 99],
        },
    }

    dcfg = DGP_DATA[dgp]
    flat_max = dcfg['flat_max']
    hier_max = dcfg['hier_max']
    max_gen = dcfg['max_gen']

    # Test set at full difficulty (final metric)
    gen_kw = {dcfg['key']: 1, dcfg['max_key']: dcfg['full_max']}
    if dcfg['ops']:
        gen_kw['ops'] = dcfg['ops']
    test_data = dcfg['gen_fn'](dcfg['n_eval'], seed=2, **gen_kw)

    # Val set: matches curriculum stage difficulty so ES is meaningful
    # (rebuilt in advance_curriculum for curriculum DGPs; full difficulty for addition)
    val_data = dcfg['gen_fn'](dcfg['n_eval'], seed=1, **gen_kw)

    # Training data: generate at full difficulty for non-curriculum, or staged later
    train_data = dcfg['gen_fn'](dcfg['n_train'], seed=0, **gen_kw)

    print(f"DGP: {dgp} ({len(train_data)} train, {len(val_data)} val, {len(test_data)} test)")

    # Hyperparams: addition stays small, mixed/algebraic get a bigger model + more epochs
    if dgp == "addition":
        d_model = 64
        nhead = 4
        num_layers = 4
        d_ff = 256
        max_epochs_default = 50
    else:
        d_model = 192
        nhead = 6
        num_layers = 5
        d_ff = 768
        max_epochs_default = 300

    max_epochs = args.max_epochs if args.max_epochs is not None else max_epochs_default
    lr = args.lr if args.lr is not None else 3e-4
    batch_size = 128
    patience = 15
    es_min_delta = 0.005  # must improve by at least 0.5% to reset patience

    model_max = 28

    # Variant configurations
    variant_configs = [
        {
            'name': 'L2R',
            'build_fn': build_flat_sequence,
            'ds_cls': FlatDataset,
            'model_cls': ArithmeticTransformer,
            'is_hier': False, 'is_hattn': False, 'is_r2l': False,
            'max_seq': flat_max,
        },
        {
            'name': 'L2R+HTP',
            'build_fn': build_hierarchical_sequence,
            'ds_cls': HierDataset,
            'model_cls': ArithmeticTransformer,
            'is_hier': True, 'is_hattn': False, 'is_r2l': False,
            'max_seq': hier_max,
        },
        {
            'name': 'L2R+HTP+HAttn',
            'build_fn': build_hierarchical_sequence,
            'ds_cls': HierDataset,
            'model_cls': HierAttnTransformer,
            'is_hier': True, 'is_hattn': True, 'is_r2l': False,
            'max_seq': hier_max,
        },
        {
            'name': 'R2L',
            'build_fn': build_r2l_flat_sequence,
            'ds_cls': FlatDataset,
            'model_cls': ArithmeticTransformer,
            'is_hier': False, 'is_hattn': False, 'is_r2l': True,
            'max_seq': flat_max,
        },
        {
            'name': 'R2L+HTP',
            'build_fn': build_r2l_hier_sequence,
            'ds_cls': HierDataset,
            'model_cls': ArithmeticTransformer,
            'is_hier': True, 'is_hattn': False, 'is_r2l': True,
            'max_seq': hier_max,
        },
        {
            'name': 'R2L+HTP+HAttn',
            'build_fn': build_r2l_hier_sequence,
            'ds_cls': HierDataset,
            'model_cls': HierAttnTransformer,
            'is_hier': True, 'is_hattn': True, 'is_r2l': True,
            'max_seq': hier_max,
        },
    ]

    # Curriculum: list of max_operand stages. ES triggers advance to next stage.
    curriculum = dcfg['curriculum']
    if curriculum:
        curriculum_stages = list(curriculum)  # e.g. [25, 50, 99]
        print(f"Curriculum stages: {curriculum_stages} (ES advances to next stage)")
    else:
        curriculum_stages = None
    current_stage_idx = 0

    def rebuild_train_dataloaders(data, configs, vlist):
        """Rebuild training dataloaders for all variants from new data."""
        for cfg, v in zip(configs, vlist):
            ds = cfg['ds_cls'](data, max_seq_len=cfg['max_seq'], build_fn=cfg['build_fn'])
            v['train_dl'] = DataLoader(
                ds, batch_size=batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(2),
            )

    def rebuild_val_dataloaders(data, configs, vlist):
        """Rebuild validation dataloaders for all variants from new data."""
        for cfg, v in zip(configs, vlist):
            ds = cfg['ds_cls'](data, max_seq_len=cfg['max_seq'], build_fn=cfg['build_fn'])
            v['val_dl'] = DataLoader(ds, batch_size=batch_size)

    def advance_curriculum(stage_idx, epoch):
        """Generate train + val data for the given curriculum stage."""
        max_op = curriculum_stages[stage_idx]
        stage_kw = {dcfg['key']: 1, dcfg['max_key']: max_op}
        if dcfg['ops']:
            stage_kw['ops'] = dcfg['ops']
        train = dcfg['gen_fn'](dcfg['n_train'], seed=epoch, **stage_kw)
        val = dcfg['gen_fn'](dcfg['n_eval'], seed=epoch + 1000, **stage_kw)
        rebuild_train_dataloaders(train, variant_configs, variants)
        rebuild_val_dataloaders(val, variant_configs, variants)
        remaining = max_epochs - epoch + 1
        is_final_stage = (stage_idx == len(curriculum_stages) - 1)
        stage_patience = 30 if is_final_stage else patience
        for v in variants:
            v['patience_left'] = stage_patience
            v['cur_patience'] = stage_patience
            v['best_acc'] = -1.0
            v['es_baseline'] = -1.0
            v['stopped'] = False
            v['stop_epoch'] = None
            v['scheduler'] = CosineAnnealingLR(v['optimizer'], T_max=remaining, eta_min=1e-5)
        print(f"  [curriculum] stage {stage_idx+1}/{len(curriculum_stages)}, max_op={max_op}, fresh LR schedule over {remaining} epochs")
        return train

    # Build datasets, models, optimizers for each variant
    variants = []
    for cfg in variant_configs:
        torch.manual_seed(2)  # identical initialization for all models

        train_ds = cfg['ds_cls'](train_data, max_seq_len=cfg['max_seq'], build_fn=cfg['build_fn'])
        val_ds = cfg['ds_cls'](val_data, max_seq_len=cfg['max_seq'], build_fn=cfg['build_fn'])
        test_ds = cfg['ds_cls'](test_data, max_seq_len=cfg['max_seq'], build_fn=cfg['build_fn'])

        model = cfg['model_cls'](
            VOCAB_SIZE, d_model=d_model, nhead=nhead, num_layers=num_layers,
            d_ff=d_ff, max_seq_len=model_max,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

        variants.append({
            'name': cfg['name'],
            'is_hier': cfg['is_hier'],
            'is_hattn': cfg['is_hattn'],
            'is_r2l': cfg['is_r2l'],
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'train_dl': DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(2),
            ),
            'val_dl': DataLoader(val_ds, batch_size=batch_size),
            'test_ds': test_ds,
            'best_acc': -1.0,
            'best_state': None,
            'es_baseline': -1.0,
            'patience_left': patience,
            'stopped': False,
            'stop_epoch': None,
        })

    n_params = sum(p.numel() for p in variants[0]['model'].parameters())
    print(f"d_model={d_model}, nhead={nhead}, layers={num_layers}, d_ff={d_ff}")
    print(f"Parameters per model: {n_params:,}")
    print(f"max_epochs={max_epochs}, lr={lr}, patience={patience}, cosine LR -> 1e-5")
    print(f"Variants: {len(variants)}")
    print()

    criterion = nn.CrossEntropyLoss()

    # History for plotting (per-variant, since they may stop at different epochs)
    history = {v['name']: {'epoch': [], 'loss': [], 'acc': []} for v in variants}

    # Checkpoint directory for this DGP
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', dgp)
    os.makedirs(ckpt_dir, exist_ok=True)
    curves_csv_path = os.path.join(ckpt_dir, 'training_curves.csv')
    results_csv_path = os.path.join(ckpt_dir, 'results.csv')

    # Resume from checkpoint if one exists
    ckpt_path = os.path.join(ckpt_dir, 'training_state.pt')
    start_epoch = 1
    if os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        start_epoch = ckpt['epoch'] + 1
        current_stage_idx = ckpt['current_stage_idx']
        history = ckpt['history']
        for v, vs in zip(variants, ckpt['variants']):
            v['model'].load_state_dict(vs['model_state'])
            v['optimizer'].load_state_dict(vs['optimizer_state'])
            v['scheduler'].load_state_dict(vs['scheduler_state'])
            v['best_acc'] = vs['best_acc']
            v['best_state'] = vs['best_state']
            v['es_baseline'] = vs.get('es_baseline', vs['best_acc'])
            v['patience_left'] = vs['patience_left']
            v['stopped'] = vs['stopped']
            v['stop_epoch'] = vs['stop_epoch']
        # Rebuild dataloaders for the current curriculum stage
        if curriculum_stages:
            advance_curriculum(current_stage_idx, epoch=start_epoch - 1)
            # Restore stopped/patience state that advance_curriculum reset
            for v, vs in zip(variants, ckpt['variants']):
                v['best_acc'] = vs['best_acc']
                v['es_baseline'] = vs.get('es_baseline', vs['best_acc'])
                v['patience_left'] = vs['patience_left']
                v['stopped'] = vs['stopped']
                v['stop_epoch'] = vs['stop_epoch']
        print(f"Resumed at epoch {start_epoch}, curriculum stage {current_stage_idx + 1}")
        # Write training_curves.csv from loaded history so CSV is complete
        rows = []
        for v in variants:
            for ep, loss, acc in zip(
                history[v['name']]['epoch'],
                history[v['name']]['loss'],
                history[v['name']]['acc'],
            ):
                rows.append((ep, v['name'], loss, acc))
        rows.sort(key=lambda x: (x[0], x[1]))
        with open(curves_csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'variant', 'loss', 'val_acc'])
            for r in rows:
                w.writerow(r)
    else:
        # Initialize first curriculum stage
        if curriculum_stages:
            train_data = advance_curriculum(current_stage_idx, epoch=0)

    def save_checkpoint(epoch):
        state = {
            'epoch': epoch,
            'current_stage_idx': current_stage_idx,
            'history': history,
            'variants': [{
                'model_state': v['model'].state_dict(),
                'optimizer_state': v['optimizer'].state_dict(),
                'scheduler_state': v['scheduler'].state_dict(),
                'best_acc': v['best_acc'],
                'best_state': v['best_state'],
                'es_baseline': v['es_baseline'],
                'patience_left': v['patience_left'],
                'stopped': v['stopped'],
                'stop_epoch': v['stop_epoch'],
            } for v in variants],
        }
        torch.save(state, ckpt_path)

    if not os.path.exists(curves_csv_path):
        with open(curves_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'variant', 'loss', 'val_acc'])

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()

        active = [v for v in variants if not v['stopped']]
        if not active:
            # All variants hit ES -- advance curriculum or stop
            if curriculum_stages and current_stage_idx < len(curriculum_stages) - 1:
                current_stage_idx += 1
                train_data = advance_curriculum(current_stage_idx, epoch)
                active = variants  # all back in play
            else:
                print(f"All variants stopped at epoch {epoch - 1}.")
                break

        for v in active:
            loss = train_epoch(
                v['model'], v['train_dl'], v['optimizer'], criterion, device,
                is_hier=v['is_hier'], is_hattn=v['is_hattn'],
                desc=f"Epoch {epoch} {v['name']}",
            )
            acc = evaluate_tf(
                v['model'], v['val_dl'], device,
                is_hier=v['is_hier'], is_hattn=v['is_hattn'],
            )

            history[v['name']]['epoch'].append(epoch)
            history[v['name']]['loss'].append(loss)
            history[v['name']]['acc'].append(acc)

            v['last_loss'] = loss
            v['last_acc'] = acc

            # Always track the actual best model
            if acc > v['best_acc']:
                v['best_acc'] = acc
                v['best_state'] = {k: val.clone() for k, val in v['model'].state_dict().items()}
                safe_name = v['name'].replace('+', '_')
                torch.save(v['best_state'], os.path.join(ckpt_dir, f"{safe_name}_best.pt"))

            # Early stopping: only reset patience on meaningful improvement
            if acc > v['es_baseline'] + es_min_delta:
                v['es_baseline'] = acc
                v['patience_left'] = v.get('cur_patience', patience)
            else:
                v['patience_left'] -= 1
                if v['patience_left'] <= 0:
                    v['stopped'] = True
                    v['stop_epoch'] = epoch

            v['scheduler'].step()

        elapsed = time.time() - t0
        n_active = sum(1 for v in variants if not v['stopped'])
        n_stopped = len(variants) - n_active
        print(f"Epoch {epoch:3d} ({elapsed:.1f}s) [{n_active} active, {n_stopped} stopped]")
        for v in active:
            tag = " [stopped]" if v['stopped'] and v['stop_epoch'] == epoch else ""
            print(f"  {v['name']:20s}  loss={v['last_loss']:.4f}  acc={v['last_acc']:.4f}{tag}")

        save_checkpoint(epoch)

        with open(curves_csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            for v in variants:
                if history[v['name']]['epoch']:
                    w.writerow([
                        epoch,
                        v['name'],
                        history[v['name']]['loss'][-1],
                        history[v['name']]['acc'][-1],
                    ])

    # Load best states for final eval
    for v in variants:
        if v['best_state'] is not None:
            v['model'].load_state_dict(v['best_state'])
        safe_name = v['name'].replace('+', '_')
        path = os.path.join(ckpt_dir, f"{safe_name}_best.pt")
        torch.save(v['model'].state_dict(), path)
    print(f"\nBest checkpoints saved to {ckpt_dir}/")

    # Run autoregressive eval on test set
    print("\nTest results (autoregressive, held-out):")
    test_accs = {}
    for v in variants:
        acc = evaluate_ar(
            v['model'], v['test_ds'], device,
            is_hier=v['is_hier'], is_hattn=v['is_hattn'],
            max_gen=max_gen, desc=f"AR eval {v['name']}",
        )
        test_accs[v['name']] = acc
        print(f"  {v['name']:20s}  exact-match accuracy: {acc:.4f}")

    with open(results_csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'best_val_acc', 'test_acc'])
        for v in variants:
            w.writerow([v['name'], v['best_acc'], test_accs[v['name']]])

    # Show examples for each variant
    for v in variants:
        print(f"\n{v['name']} examples (test set):")
        show_examples(
            v['model'], v['test_ds'], device,
            is_hier=v['is_hier'], is_hattn=v['is_hattn'], is_r2l=v['is_r2l'],
            max_gen=max_gen,
        )

    # Save plots
    plot_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    save_plots(history, plot_dir, dgp_name=dgp)


if __name__ == '__main__':
    main()
