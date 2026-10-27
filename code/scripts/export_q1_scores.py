import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR
from src.q1.data import load_q1

def main():
    print("生成最终的宽表 (q1_final_scores.csv)...")
    data = load_q1()
    scores = np.load(PROCESSED_DIR / "q1_scores.npz", allow_pickle=True)
    G_hat = scores['G_hat']
    
    df = pd.DataFrame(G_hat, columns=data.dims)
    df.insert(0, "beer", data.beer)
    df.to_csv(PROCESSED_DIR / "q1_final_scores.csv", index=False)
    print("✅ 宽表已生成。")

if __name__ == "__main__":
    main()