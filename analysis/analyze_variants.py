import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from bionemo.esm2.model.finetune.token_classifier import ESM2FineTuneSeqModel
import numpy as np
import os

# --- CONFIGURATION ---
CHECKPOINT_DIR = "/workspace/project/results/run_centralized_human/checkpoints"

#  REPLACE THESE WITH REAL SEQUENCES FROM YOUR DATA 
# I have put placeholders here. needto copy-paste real strings from your CSV.
VARIANTS = [
    ("Wild Type", "M...PUT_REAL_SEQUENCE_HERE..."), 
    ("Mutant A",  "M...PUT_REAL_SEQUENCE_HERE..."),
    ("Mutant B",  "M...PUT_REAL_SEQUENCE_HERE..."),
    ("Double Mut","M...PUT_REAL_SEQUENCE_HERE..."),
]

def get_checkpoint_path():
    last = os.path.join(CHECKPOINT_DIR, "last.ckpt")
    if os.path.exists(last): return last
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.nemo')]
    if files: return os.path.join(CHECKPOINT_DIR, files[0])
    raise FileNotFoundError("No checkpoint found.")

def get_embedding(model, sequence):
    tokens = model.tokenizer.text_to_ids(sequence)
    tokens = torch.tensor([tokens]).cuda()
    lengths = torch.tensor([len(tokens[0])]).cuda()
    
    with torch.no_grad():
        output = model(tokens, lengths)
        # Get last hidden state and mean pool
        last_hidden = output['encoder_hidden_states'][-1] 
        embedding = last_hidden.mean(dim=1).squeeze().cpu().numpy()
        
    return embedding

def main():
    ckpt = get_checkpoint_path()
    print(f"  Loading model: {ckpt}")
    model = ESM2FineTuneSeqModel.load_from_checkpoint(ckpt)
    model.eval().cuda()
    
    embeddings = []
    labels = []
    
    print("🧬 Extracting embeddings for variants...")
    # Validate sequences aren't placeholders
    valid_variants = []
    for label, seq in VARIANTS:
        if "PUT_REAL_SEQUENCE" in seq:
            print(f" Skipping {label}: You didn't update the sequence string!")
            continue
        valid_variants.append((label, seq))

    if not valid_variants:
        print(" No valid sequences found. Please edit VARIANTS list in the script.")
        return

    for label, seq in valid_variants:
        emb = get_embedding(model, seq)
        embeddings.append(emb)
        labels.append(label)
        
    # PCA to 2D
    # We need at least 2 samples to run PCA
    if len(embeddings) < 2:
        print(" Need at least 2 variants to plot.")
        return

    pca = PCA(n_components=2)
    # If we have very few samples, PCA might complain, but usually handles it
    reduced = pca.fit_transform(np.array(embeddings))
    
    # Plot
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        x, y = reduced[i]
        plt.scatter(x, y, s=200, label=label)
        plt.text(x+0.05, y+0.05, label, fontsize=12)

    plt.title("Latent Space: Variant Comparison")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("variant_embeddings.png")
    print(f"Saved visualization to variant_embeddings.png")

if __name__ == "__main__":
    main()