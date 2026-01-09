import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from bionemo.esm2.model.finetune.token_classifier import ESM2FineTuneSeqModel
import os

# Configuration
# Check if "last.ckpt" exists, otherwise try to find the best one
CHECKPOINT_DIR = "/workspace/project/results/run_centralized_human/checkpoints"
TEST_DATA_PATH = "/workspace/project/data/splits/human/val.csv" # Using Val for now as Test might not exist
OUTPUT_IMG = "evaluation_metrics.png"

def get_checkpoint_path():
    # Prefer 'last.ckpt'
    last = os.path.join(CHECKPOINT_DIR, "last.ckpt")
    if os.path.exists(last):
        return last
    # Fallback: Find the .nemo file (best checkpoint)
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.nemo')]
    if files:
        return os.path.join(CHECKPOINT_DIR, files[0])
    raise FileNotFoundError("Could not find a valid checkpoint (last.ckpt or *.nemo)")

def evaluate():
    ckpt_path = get_checkpoint_path()
    print(f"⬇  Loading model from: {ckpt_path}")
    
    # Load model
    model = ESM2FineTuneSeqModel.load_from_checkpoint(ckpt_path)
    model.eval()
    model.cuda()

    print(f"Loading evaluation data from: {TEST_DATA_PATH}")
    df = pd.read_csv(TEST_DATA_PATH)
    
    # Validation
    required_cols = ['sequences', 'DMS_score']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    predictions = []
    ground_truth = df['DMS_score'].values
    
    print(f"Running Inference on {len(df)} sequences...")
    with torch.no_grad():
        for i, seq in enumerate(df['sequences']):
            if i % 100 == 0: print(f"   ... processing {i}/{len(df)}")
            
            # Tokenize
            tokens = model.tokenizer.text_to_ids(seq)
            tokens = torch.tensor([tokens]).cuda()
            lengths = torch.tensor([len(tokens[0])]).cuda()
            
            # Forward pass
            output = model(tokens, lengths)
            pred = output['regression_output'].item()
            predictions.append(pred)

    predictions = np.array(predictions)

    # Metrics
    pears, _ = pearsonr(ground_truth, predictions)
    spear, _ = spearmanr(ground_truth, predictions)
    mse = mean_squared_error(ground_truth, predictions)

    print("\n" + "="*30)
    print(" FINAL METRICS")
    print("="*30)
    print(f"Pearson (r):      {pears:.4f}")
    print(f"Spearman (rho):   {spear:.4f}")
    print(f"MSE:              {mse:.4f}")
    print("="*30)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.regplot(x=ground_truth, y=predictions, scatter_kws={'alpha':0.4, 'color': 'blue'}, line_kws={'color':'red'})
    plt.title(f"Predicted vs True DMS Score\nSpearman: {spear:.3f} | Pearson: {pears:.3f}")
    plt.xlabel("True DMS Score (Ground Truth)")
    plt.ylabel("Predicted Score")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUTPUT_IMG)
    print(f"Saved plot image to: {os.path.abspath(OUTPUT_IMG)}")

if __name__ == "__main__":
    evaluate()