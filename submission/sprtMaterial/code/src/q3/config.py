import numpy as np

TAU = 0.6
N_REPEATS = 5
N_FOLDS = 5
N_SYNTH = 1000
N_BOOT = 2000
N_PERM = 200
GRID_K = (2, 5, 10, 15, 20, 25, 30)
GRID_KAPPA = (0, 1, 2, 5, 10, 20)
ALPHA_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
FLOOR = -1.81781854
SMALL_STYLES = ("Faro", "Brut", "Scotch", "Flanders old brown")
SRC_STYLE = "Pils/Lager"
TGT_STYLE = "Tripel"
NAMED_STYLES = ("Tripel", "Kriek", "Lambic")
DEFECT_CHEM = (
    "acetaldehyde",
    "DES",
    "DMS",
    "H2S",
    "MeSH",
    "SO2.mg.L.",
    "X2.3.butanedione",
    "X4VG",
    "X.E..2.nonenal",
)
DEFECT_SENS = ("dms", "metallic", "X4vg", "t2n", "diacetyl", "stale_hops")
COMPOSITE_COLS = ("esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum", "kcalperc")
INTERNAL_STD_COL = "X3.methyl.2.butenal"
