"""Export training_curves.csv and results.csv from an existing training_state.pt.

Usage:
  python export_training_csvs.py checkpoints/mixed
  python export_training_csvs.py checkpoints/mixed --log htp/mixed_extended.log

If --log is given, parses the log for curriculum stage 3 and prints max val acc after it.
"""

import argparse
import csv
import os
import re

import torch


def main():
    parser = argparse.ArgumentParser(description="Export CSVs from training_state.pt")
    parser.add_argument("ckpt_dir", help="Checkpoint dir containing training_state.pt")
    parser.add_argument("--log", default=None, help="Optional log file to parse for max acc after stage 3")
    args = parser.parse_args()

    state_path = os.path.join(args.ckpt_dir, "training_state.pt")
    if not os.path.exists(state_path):
        print(f"Not found: {state_path}")
        return

    ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
    history = ckpt["history"]
    variants_state = ckpt["variants"]

    variant_names = list(history.keys())
    curves_path = os.path.join(args.ckpt_dir, "training_curves.csv")
    rows = []
    for name in variant_names:
        for ep, loss, acc in zip(
            history[name]["epoch"],
            history[name]["loss"],
            history[name]["acc"],
        ):
            rows.append((ep, name, loss, acc))
    rows.sort(key=lambda x: (x[0], x[1]))
    with open(curves_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "variant", "loss", "val_acc"])
        for r in rows:
            w.writerow(r)
    print(f"Wrote {curves_path} ({len(rows)} rows)")

    results_path = os.path.join(args.ckpt_dir, "results.csv")
    with open(results_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "best_val_acc", "test_acc"])
        for name, vs in zip(variant_names, variants_state):
            w.writerow([name, vs["best_acc"], ""])
    print(f"Wrote {results_path} (test_acc left blank; run eval.py for test metrics)")

    best_overall = max(vs["best_acc"] for vs in variants_state)
    print(f"Best validation accuracy (overall, from checkpoints): {best_overall:.4f}")

    if args.log and os.path.exists(args.log):
        with open(args.log) as f:
            text = f.read()
        idx = text.find("stage 3/3")
        if idx >= 0:
            after = text[idx:]
            accs = re.findall(r"acc=([0-9.]+)", after)
            if accs:
                max_acc = max(float(a) for a in accs)
                print(f"Max validation accuracy after curriculum stage 3 (from log): {max_acc:.4f}")
            else:
                print("(Log has stage 3/3 but no acc= after it, e.g. resumed run; use best overall above.)")
        else:
            print("Log does not contain 'stage 3/3' (no curriculum or stage 3 not reached)")


if __name__ == "__main__":
    main()
