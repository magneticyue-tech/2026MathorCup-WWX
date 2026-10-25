import numpy as np
from src.q1.panels import panel_zeros
import pandas as pd

def test_panel_zeros_logic():
    # 构造一个小样例：3款酒，1个维度
    Z = pd.DataFrame({"dim1": [0.1, 0.1, 0.5]})
    panels = np.array([0, 0, 0]) # 都在同一个组
    
    zeros_df = panel_zeros(Z, panels)
    # 最小的 0.1 出现了两次，应该被识别为零点
    assert zeros_df.shape[1] == 1
    assert not np.isnan(zeros_df.iloc[0, 0]), "样例验证失败：未能识别出并列最小值零点"
    print("✅ 单元测试通过：零点识别逻辑正确。")

if __name__ == "__main__":
    test_panel_zeros_logic()
