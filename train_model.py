"""Train both models from the command line (same thing as the 'Train models' page).

    python train_model.py                      # bundled dataset, original settings
    python train_model.py --csv my.csv --target Label --k 30
"""
import argparse
import ml_pipeline as ml

p = argparse.ArgumentParser()
p.add_argument("--csv", default=ml.DEFAULT_DATASET)
p.add_argument("--target", default=ml.DEFAULT_TARGET)
p.add_argument("--k", type=int, default=20, help="features to keep (SelectKBest)")
p.add_argument("--test-size", type=float, default=0.2)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--no-scale", action="store_true", help="do not standardise features for the neural network")
a = p.parse_args()

df = ml.load_dataset(a.csv)
print(f"Loaded {len(df)} rows. Training ...")
m = ml.train_all(df, a.target, a.k, a.test_size, not a.no_scale, a.epochs, ml.STATE["dataset_name"],
                 progress=lambda i: print(f"  step {i + 1}/{len(ml.STEPS)}: {ml.STEPS[min(i, len(ml.STEPS) - 1)]}"))
for name in ("adaboost", "dnn"):
    r = m[name]
    print(f"{name:9s} acc={r['accuracy']:.4f} prec={r['precision']:.4f} rec={r['recall']:.4f} "
          f"f1={r['f1']:.4f} fpr={r['fpr']:.4f}")
print("Saved to the models/ folder.")
