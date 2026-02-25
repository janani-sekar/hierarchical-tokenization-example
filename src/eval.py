"""Evaluate saved checkpoints with autoregressive decoding and plot results."""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from data import generate_addition_dataset, generate_mixed_dataset, generate_algebraic_dataset
from tokenizer import (
    VOCAB_SIZE,
    build_flat_sequence, build_r2l_flat_sequence,
    build_hierarchical_sequence, build_r2l_hier_sequence,
)
from model import ArithmeticTransformer, HierAttnTransformer
from train import FlatDataset, HierDataset, evaluate_ar, VARIANT_STYLE


DGP_CONFIGS = {
    'addition': {
        'gen_fn': lambda seed: generate_addition_dataset(5_000, min_val=1, max_val=999, seed=seed),
        'flat_max': 17,
        'hier_max': 19,
        'max_gen': 5,
    },
    'mixed': {
        'gen_fn': lambda seed: generate_mixed_dataset(5_000, min_val=1, max_val=99, ops="+-*/", seed=seed),
        'flat_max': 15,
        'hier_max': 17,
        'max_gen': 7,
    },
    'algebraic': {
        'gen_fn': lambda seed: generate_algebraic_dataset(5_000, min_coeff=1, max_coeff=99, ops="+-*/", seed=seed),
        'flat_max': 19,
        'hier_max': 21,
        'max_gen': 8,
    },
}


def _variant_configs(flat_max, hier_max):
    return [
        {
            'name': 'L2R',
            'build_fn': build_flat_sequence,
            'ds_cls': FlatDataset,
            'model_cls': ArithmeticTransformer,
            'is_hier': False, 'is_hattn': False,
            'max_seq': flat_max,
        },
        {
            'name': 'L2R+HTP',
            'build_fn': build_hierarchical_sequence,
            'ds_cls': HierDataset,
            'model_cls': ArithmeticTransformer,
            'is_hier': True, 'is_hattn': False,
            'max_seq': hier_max,
        },
        {
            'name': 'L2R+HTP+HAttn',
            'build_fn': build_hierarchical_sequence,
            'ds_cls': HierDataset,
            'model_cls': HierAttnTransformer,
            'is_hier': True, 'is_hattn': True,
            'max_seq': hier_max,
        },
        {
            'name': 'R2L',
            'build_fn': build_r2l_flat_sequence,
            'ds_cls': FlatDataset,
            'model_cls': ArithmeticTransformer,
            'is_hier': False, 'is_hattn': False,
            'max_seq': flat_max,
        },
        {
            'name': 'R2L+HTP',
            'build_fn': build_r2l_hier_sequence,
            'ds_cls': HierDataset,
            'model_cls': ArithmeticTransformer,
            'is_hier': True, 'is_hattn': False,
            'max_seq': hier_max,
        },
        {
            'name': 'R2L+HTP+HAttn',
            'build_fn': build_r2l_hier_sequence,
            'ds_cls': HierDataset,
            'model_cls': HierAttnTransformer,
            'is_hier': True, 'is_hattn': True,
            'max_seq': hier_max,
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints on arithmetic tasks.")
    parser.add_argument("--dgp", choices=list(DGP_CONFIGS.keys()), default="addition",
                        help="Data generating process (default: addition)")
    args = parser.parse_args()

    dgp = args.dgp
    dgp_cfg = DGP_CONFIGS[dgp]

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f"Device: {device}")
    print(f"DGP: {dgp}")

    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', dgp)
    plot_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    if dgp == 'addition':
        d_model, nhead, num_layers, d_ff = 64, 4, 4, 256
    else:
        d_model, nhead, num_layers, d_ff = 192, 6, 5, 768

    test_data = dgp_cfg['gen_fn'](seed=2)
    max_gen = dgp_cfg['max_gen']
    model_max = 28
    results = {}

    for cfg in _variant_configs(dgp_cfg['flat_max'], dgp_cfg['hier_max']):
        safe_name = cfg['name'].replace('+', '_')
        ckpt_path = os.path.join(ckpt_dir, f"{safe_name}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  {cfg['name']:20s}  checkpoint not found, skipping")
            continue

        model = cfg['model_cls'](
            VOCAB_SIZE, d_model=d_model, nhead=nhead, num_layers=num_layers,
            d_ff=d_ff, max_seq_len=model_max,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        test_ds = cfg['ds_cls'](
            test_data, max_seq_len=cfg['max_seq'], build_fn=cfg['build_fn'],
        )

        acc = evaluate_ar(
            model, test_ds, device,
            is_hier=cfg['is_hier'], is_hattn=cfg['is_hattn'],
            max_gen=max_gen, desc=f"AR eval {cfg['name']}",
        )
        results[cfg['name']] = acc
        print(f"  {cfg['name']:20s}  exact-match accuracy: {acc:.4f}")

    if not results:
        print("No checkpoints found. Run train.py first.")
        return

    # Bar chart
    names = list(results.keys())
    accs = [results[n] for n in names]
    colors = [VARIANT_STYLE[n]['color'] for n in names]
    hatches = ['///' if VARIANT_STYLE[n]['ls'] == '--' else '' for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(names)), accs, color=colors, edgecolor='black', linewidth=0.5)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Exact-match Accuracy')
    ax.set_title('Test Accuracy (autoregressive decoding)')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    for i, acc in enumerate(accs):
        ax.text(i, acc + 0.01, f"{acc:.2%}", ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    suffix = f'_{dgp}' if dgp else ''
    path = os.path.join(plot_dir, f'test_accuracy{suffix}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {path}")


if __name__ == '__main__':
    main()
