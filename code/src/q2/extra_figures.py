# code/src/q2/extra_figures.py
"""Q2 辅助图表生成：网络图、方法对比图、稳定性分布图、模块大小图。"""
import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.manifold import MDS
import config as C

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_network_graph():
    print("正在绘制共表达网络图...")
    df_nodes = pd.read_csv(C.Q2_DIR / "graph_nodes.csv")
    df_edges = pd.read_csv(C.Q2_DIR / "graph_edges.csv")
    
    G = nx.Graph()
    for _, row in df_nodes.iterrows():
        G.add_node(row["node"], module=row["module_id"], k_in=row["k_in"], is_cand=row["is_candidate"])
    for _, row in df_edges.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["weight_abs"])

    # 提取最大连通子图计算 MDS
    largest_cc = max(nx.connected_components(G), key=len)
    G_sub = G.subgraph(largest_cc)
    nodes = list(G_sub.nodes())
    n = len(nodes)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        lengths = nx.single_source_dijkstra_path_length(G_sub, nodes[i], weight=lambda u, v, d: 1.0 - d['weight'] + 0.01)
        for j in range(n):
            dist_matrix[i, j] = lengths.get(nodes[j], 2.0)
            
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=C.SEED, n_init=1, max_iter=100)
    coords = mds.fit_transform(dist_matrix)
    pos = {node: coords[i] for i, node in enumerate(nodes)}
    for node in G.nodes():
        if node not in pos: pos[node] = np.random.randn(2) * 2

    plt.figure(figsize=(14, 10))
    modules = df_nodes["module_id"].unique()
    cmap = plt.get_cmap("tab20", len(modules))
    module_colors = {m: cmap(i) for i, m in enumerate(modules)}
    
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='gray')
    node_colors = [module_colors[G.nodes[node]["module"]] for node in G.nodes()]
    node_sizes = [50 + G.nodes[node]["k_in"] * 150 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85, edgecolors='white')
    
    # 只给前 10 个 Hub 节点加标签
    hubs = df_nodes.sort_values("k_in", ascending=False).head(10)["node"].tolist()
    for node in hubs:
        x, y = pos[node]
        plt.text(x, y + 0.1, node, fontsize=9, ha='center', va='bottom',
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.3))

    plt.title(r"Compound Co-expression Network (\(\tau = 0.6\))", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(C.FIGURES / "network_graph.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_method_jaccard():
    print("正在绘制方法对比柱状图...")
    # 读取敏感性分析（M0/M1/tau）和高级控制（M2/M3/风格内）
    sens_file = C.Q2_DIR / "controls_sensitivity_results.csv"
    full_file = C.Q2_DIR / "full_controls_results.csv"
    
    if not sens_file.exists() or not full_file.exists():
        print("  ⚠️ 找不到对比结果表，跳过。")
        return
        
    df_sens = pd.read_csv(sens_file)
    df_full = pd.read_csv(full_file)
    
    data = [
        ("M0 (无图边际)", df_sens["jaccard_M0"].mean()),
        ("M1 (图约束边际)", df_sens["jaccard_M1"].mean()),
        ("M2 (ElasticNet)", df_full["jaccard_M2"].mean()),
        ("M3 (GBM)", df_full["jaccard_M3"].mean()),
        ("风格内", df_full["jaccard_Within"].mean())
    ]
    labels, values = zip(*data)
    
    plt.figure(figsize=(8, 5))
    bars = plt.barh(labels, values, color=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
    
    for bar in bars:
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}', va='center')
        
    plt.xlabel("平均 Jaccard 相似度 (与主方法相比)")
    plt.title("不同对照方法 vs 主方法 (49维平均)")
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(C.FIGURES / "method_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_stability_hist():
    print("正在绘制稳定性入选频率分布图...")
    df = pd.read_csv(C.Q2_DIR / "final_top3_with_stability.csv")
    pi_c_values = df["pi_c"].dropna().values
    
    plt.figure(figsize=(8, 5))
    sns.histplot(pi_c_values, bins=20, kde=True, color="coral")
    plt.axvline(0.6, color='red', linestyle='--', label='稳定门槛 (0.6)')
    plt.axvline(1/3, color='orange', linestyle='--', label='中等门槛 (1/3)')
    plt.title("Top-3 化合物 Bootstrap 入选频率 (pi_c) 分布")
    plt.xlabel("入选频率 (pi_c)")
    plt.ylabel("频数")
    plt.legend()
    plt.tight_layout()
    plt.savefig(C.FIGURES / "stability_pi_c_hist.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_module_sizes():
    print("正在绘制模块大小分布图...")
    df_modules = pd.read_csv(C.Q2_DIR / "graph_modules.csv")
    sizes = df_modules["size"].values
    
    plt.figure(figsize=(8, 5))
    sns.histplot(sizes, bins=range(1, max(sizes)+2), discrete=True, color="skyblue")
    plt.title("模块大小分布 (τ=0.6)")
    plt.xlabel("模块内化合物数量")
    plt.ylabel("模块个数")
    plt.tight_layout()
    plt.savefig(C.FIGURES / "module_sizes.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_network_graph()
    plot_method_jaccard()
    plot_stability_hist()
    plot_module_sizes()
    print("✅ 所有辅助图表已保存至 figures/q2/")