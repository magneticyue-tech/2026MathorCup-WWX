import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.q1.paths import PROCESSED_DIR, SEED
from src.q1.data import load_q1, ChemNAEncoder
from src.q1.splits import outer_splits
from src.q1.discretize import grade, dist_median

def main():
    print("正在运行 T4 全量对照组 (B0/B1/EN/RF/K2/C1/C2)...")
    data = load_q1()
    q1_data = np.load(PROCESSED_DIR / "q1_data.npz", allow_pickle=True)
    xbar = q1_data['xbar']
    G_obs = q1_data['G']
    date_group = q1_data['date_group']
    
    main_oof = np.load(PROCESSED_DIR / "q1_main_oof.npz", allow_pickle=True)
    chosen_params = json.loads(str(main_oof['chosen_params']))
    
    encoder = ChemNAEncoder(scheme="col_mean")
    encoder.fit(data.X)
    X_proc, miss_inds = encoder.transform(data.X)
    X_proc = pd.concat([X_proc, miss_inds], axis=1)
    X_arr = X_proc.values
    Z_arr = data.Z.values 
    style_arr = data.style.values
    
    splits = outer_splits(date_group, data.style, n_splits=5, n_repeats=5, seed=SEED)
    
    models = ["B0", "B1", "EN", "RF", "K2", "C1", "C2"]
    controls_oof = {m: np.zeros((5, 250, 50)) for m in ["B0", "B1", "EN", "RF", "C2"]}
    controls_G = {m: np.zeros((5, 250, 50), dtype=int) for m in models}
    controls_p = {m: np.zeros((5, 250, 50, 7)) for m in ["K2", "C1", "C2"]}

    for run in range(5):
        for fold_idx, (train_idx, test_idx) in enumerate(splits[run]):
            global_mean = np.mean(xbar[train_idx], axis=0)
            controls_oof["B0"][run, test_idx] = global_mean
            
            train_styles = style_arr[train_idx]
            for i, idx in enumerate(test_idx):
                s = style_arr[idx]
                mask = (train_styles == s)
                n_s = mask.sum()
                if n_s > 0:
                    mean_s = np.mean(xbar[train_idx][mask], axis=0)
                    controls_oof["B1"][run, test_idx[i]] = (n_s * mean_s + 5 * global_mean) / (n_s + 5)
                else:
                    controls_oof["B1"][run, test_idx[i]] = global_mean
                    
            controls_G["B0"][run, test_idx] = grade(controls_oof["B0"][run, test_idx])
            controls_G["B1"][run, test_idx] = grade(controls_oof["B1"][run, test_idx])

            for dim_idx, dim in enumerate(data.dims):
                y_tr = xbar[train_idx, dim_idx]
                
                en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
                en.fit(X_arr[train_idx], y_tr)
                controls_oof["EN"][run, test_idx, dim_idx] = en.predict(X_arr[test_idx])
                
                rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", min_samples_leaf=5, random_state=SEED)
                rf.fit(X_arr[train_idx], y_tr)
                controls_oof["RF"][run, test_idx, dim_idx] = rf.predict(X_arr[test_idx])

            controls_G["EN"][run, test_idx] = grade(controls_oof["EN"][run, test_idx])
            controls_G["RF"][run, test_idx] = grade(controls_oof["RF"][run, test_idx])

            for dim_idx, dim in enumerate(data.dims):
                p_k2 = np.zeros((len(test_idx), 7))
                for k in range(7):
                    y_tr_k = (G_obs[train_idx, dim_idx] >= k).astype(int)
                    if y_tr_k.sum() < 2: p_k2[:, k] = y_tr_k.mean()
                    else:
                        clf = HistGradientBoostingClassifier(**chosen_params[dim], random_state=SEED)
                        clf.fit(X_arr[train_idx], y_tr_k)
                        p_k2[:, k] = clf.predict_proba(X_arr[test_idx])[:, 1]
                F = np.maximum.accumulate(p_k2, axis=1)
                p_final = np.diff(np.hstack((np.zeros((len(test_idx), 1)), F)), axis=1)
                controls_p["K2"][run, test_idx, dim_idx] = p_final
                controls_G["K2"][run, test_idx, dim_idx] = dist_median(p_final)

            for dim_idx, dim in enumerate(data.dims):
                y_z_tr = Z_arr[train_idx, dim_idx]
                c1_model = HistGradientBoostingRegressor(**chosen_params[dim], random_state=SEED)
                c1_model.fit(X_arr[train_idx], y_z_tr)
                z_pred_te = c1_model.predict(X_arr[test_idx])
                controls_G["C1"][run, test_idx, dim_idx] = grade(z_pred_te)

            for dim_idx, dim in enumerate(data.dims):
                y_z_tr = Z_arr[train_idx, dim_idx]
                y_x_tr = xbar[train_idx, dim_idx]
                c2_model = HistGradientBoostingRegressor(**chosen_params[dim], random_state=SEED)
                c2_model.fit(X_arr[train_idx], y_z_tr)
                z_pred_tr = c2_model.predict(X_arr[train_idx])
                z_pred_te = c2_model.predict(X_arr[test_idx])
                ols_coef = np.polyfit(z_pred_tr, y_x_tr, 1)
                x_pred_te = np.polyval(ols_coef, z_pred_te)
                controls_oof["C2"][run, test_idx, dim_idx] = x_pred_te
            controls_G["C2"][run, test_idx] = grade(controls_oof["C2"][run, test_idx])

    save_path = PROCESSED_DIR / "q1_controls.npz"
    save_dict = {
        **{f"oof_{m}": controls_oof[m] for m in ["B0", "B1", "EN", "RF", "C2"]},
        **{f"G_{m}": controls_G[m] for m in models},
        **{f"p_{m}": controls_p[m] for m in ["K2", "C1", "C2"]}
    }
    np.savez_compressed(save_path, **save_dict)
    print(f"✅ T4 运行完毕，保存至: {save_path}")

if __name__ == "__main__":
    main()