import numpy as np
from src.q3.data import defect_table, load_chem, load_dictionary


def test_defect_table_rows():
    dic = load_dictionary()
    chem = load_chem(dic=dic)
    assert len(chem.gen_cols) == 225
    assert len(defect_table(chem, dic)) == 9
    assert np.isnan(chem.XG).sum() == 60
