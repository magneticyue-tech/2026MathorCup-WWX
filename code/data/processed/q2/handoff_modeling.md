# Q2 结果移交（Coding → Modeling）

数值来源：`code/data/processed/q2/final_top3_with_rationality.csv`。`interpretation` 列留空，由 Modeling 撰写。

- A 级 22 个：A_malt_all、A_malt_bread、A_malt_cara、A_malt_burn、A_hops_noble、A_esters_all、A_esters_isoaa、F_malt_cara、F_hops_all、F_hops_noble、F_esters_all、F_esters_ethac、acidity、bitternes、X4vg、diacetyl、dms、orange、clove、alcohol、body、co2
- B 级 25 个：A_malt_grain、A_hops_all、A_hops_citrus、A_hops_tropical、A_hops_woody、A_esters_ethac、A_esters_fruity、F_malt_all、F_malt_grain、F_malt_bread、F_malt_burn、F_hops_citrus、F_hops_tropical、F_hops_woody、F_esters_isoaa、F_esters_flower、F_esters_fruity、sweetness、metallic、stale_hops、coriander、lactic、acetic、aftertaste、overall
- C 级 2 个：A_esters_flower、t2n

## 主方法每维 Top-3

| 维度 | 等级 | 第 1 位 | 第 2 位 | 第 3 位 |
|---|---|---|---|---|
| A_malt_all | A | color.EBC.（+，π^c=0.90，显著） | X4.ethyl.phenol（-，π^c=0.15，显著） | beta.phellandrene（-，π^c=0.19，显著） |
| A_malt_grain | B | terpinen.4.ol（-，π^c=0.36，显著） | X2.acetylfuran（+，π^c=0.10，不显著） | X.E..beta.ionone（-，π^c=0.06，不显著） |
| A_malt_bread | A | color.EBC.（+，π^c=0.65，显著） | X4.ethyl.guaiacol（-，π^c=0.16，显著） | ethyl.tiglate（+，π^c=0.16，不显著） |
| A_malt_cara | A | color.EBC.（+，π^c=0.99，显著） | X2.ethyl.1.hexanol（-，π^c=0.40，显著） | GFS（+，π^c=0.41，显著） |
| A_malt_burn | A | color.EBC.（+，π^c=0.67，显著） | DES（+，π^c=0.17，显著） | isopropyl.hexadecanoate（-，π^c=0.04，不显著） |
| A_hops_all | B | ethyl_octanoate（+，π^c=0.53，显著） | iron（-，π^c=0.28，显著） | linalool（+，π^c=0.40，显著） |
| A_hops_citrus | B | citronellol（+，π^c=0.45，显著） | CO2.PSI.（+，π^c=0.38，显著） | X6.methyl.5.hepten.2.one（+，π^c=0.24，显著） |
| A_hops_tropical | B | DMS（+，π^c=0.54，显著） | X.E..beta.ocimene（+，π^c=0.20，不显著） | methyl.geranate（+，π^c=0.10，不显著） |
| A_hops_noble | A | ethyl_octanoate（+，π^c=0.64，显著） | iron（-，π^c=0.36，显著） | X.E..2.nonenal（+，π^c=0.19，不显著） |
| A_hops_woody | B | trans.dihydro.alpha.terpinyl.acetate（+，π^c=0.08，显著） | X2Z.6Z.farnesol（+，π^c=0.19，不显著） | car.3.en.2.one（-，π^c=0.21，不显著） |
| A_esters_all | A | n.hexanol（+，π^c=0.61，显著） | lactic_acid.mg.L.（+，π^c=0.38，显著） | acetaldehyde（+，π^c=0.37，不显著） |
| A_esters_ethac | B | ethyl_acetate（+，π^c=0.37，显著） | X2.methyl.1.butanol（+，π^c=0.15，显著） | X2.3.dihydro.farnesol（+，π^c=0.09，不显著） |
| A_esters_isoaa | A | n.hexanol（+，π^c=0.73，显著） | tau.cadinol（-，π^c=0.41，显著） | ethyl_decanoate（-，π^c=0.10，不显著） |
| A_esters_flower | C | beta.phellandrene（+，π^c=0.27，不显著） | n.hexanol（+，π^c=0.13，不显著） | acetic_acid.g.L.（+，π^c=0.08，不显著） |
| A_esters_fruity | B | beta.phellandrene（+，π^c=0.50，显著） | X1.10.di.epi.cubenol（-，π^c=0.16，不显著） | neryl.formate（+，π^c=0.16，显著） |
| F_malt_all | B | ethanol..v.v.（+，π^c=0.51，显著） | color.EBC.（+，π^c=0.66，显著） | pH（+，π^c=0.55，显著） |
| F_malt_grain | B | isoborneol（-，π^c=0.19，显著） | X2.3.butanedione（-，π^c=0.17，不显著） | propyl.acetate（+，π^c=0.13，不显著） |
| F_malt_bread | B | acetaldehyde（+，π^c=0.26，显著） | X2.acetylpyrrole（+，π^c=0.35，显著） | neryl.formate（-，π^c=0.18，显著） |
| F_malt_cara | A | color.EBC.（+，π^c=0.91，显著） | acetaldehyde（+，π^c=0.55，显著） | acetic_acid.g.L.（-，π^c=0.22，显著） |
| F_malt_burn | B | ethanol..v.v.（+，π^c=0.54，显著） | pH（+，π^c=0.45，显著） | color.EBC.（+，π^c=0.55，显著） |
| F_hops_all | A | bitterness（+，π^c=0.83，显著） | ethyl_octanoate（+，π^c=0.52，显著） | linalool（+，π^c=0.24，显著） |
| F_hops_citrus | B | protein.g.L.（-，π^c=0.26，显著） | nerol.oxide（+，π^c=0.20，显著） | alpha.terpinyl.isobutanoate（-，π^c=0.09，不显著） |
| F_hops_tropical | B | geraniol（+，π^c=0.26，显著） | DMS（+，π^c=0.27，显著） | hexyl.acetate（+，π^c=0.34，不显著） |
| F_hops_noble | A | bitterness（+，π^c=0.72，显著） | ethyl_octanoate（+，π^c=0.77，显著） | phenethyl_alcohol（+，π^c=0.12，不显著） |
| F_hops_woody | B | bitterness（+，π^c=0.48，显著） | ethyl.hexadecanoate（+，π^c=0.20，显著） | isobutanol（+，π^c=0.27，不显著） |
| F_esters_all | A | protein.g.L.（+，π^c=0.71，显著） | n.hexanol（+，π^c=0.43，显著） | pH（-，π^c=0.20，显著） |
| F_esters_ethac | A | ethyl_acetate（+，π^c=0.71，显著） | E.nerolidol（+，π^c=0.18，显著） | styrene（-，π^c=0.04，不显著） |
| F_esters_isoaa | B | n.hexanol（+，π^c=0.45，显著） | protein.g.L.（+，π^c=0.39，显著） | tau.cadinol（-，π^c=0.10，不显著） |
| F_esters_flower | B | beta.phellandrene（+，π^c=0.43，显著） | isopentyl.butanoate（-，π^c=0.43，显著） | X.E..dihydro.apofarnesal（-，π^c=0.17，不显著） |
| F_esters_fruity | B | DES（-，π^c=0.48，显著） | protein.g.L.（+，π^c=0.53，显著） | beta.phellandrene（+，π^c=0.43，显著） |
| acidity | A | pH（-，π^c=0.85，显著） | isopentyl_acetate（-，π^c=0.19，显著） | eugenol（+，π^c=0.09，不显著） |
| bitternes | A | bitterness（+，π^c=1.00，显著） | ethyl_octanoate（+，π^c=0.80，显著） | CO2.PSI.（+，π^c=0.20，显著） |
| sweetness | B | ethyl_decanoate（-，π^c=0.49，显著） | protein.g.L.（+，π^c=0.50，显著） | unfermentables（+，π^c=0.31，显著） |
| X4vg | A | X4VG（+，π^c=0.94，显著） | neryl.formate（-，π^c=0.41，显著） | betaglucan.mg.L.（+，π^c=0.18，显著） |
| diacetyl | A | eugenol（-，π^c=0.66，显著） | furfural（+，π^c=0.19，显著） | phenethyl_acetate（+，π^c=0.05，不显著） |
| dms | A | protein.g.L.（-，π^c=0.76，显著） | terpinen.4.ol（-，π^c=0.29，不显著） | unfermentables（-，π^c=0.21，不显著） |
| metallic | B | ethyl_isovalerate（-，π^c=0.31，显著） | iron（+，π^c=0.62，显著） | ethyl_octanoate（+，π^c=0.08，显著） |
| stale_hops | B | X2.undecanol（+，π^c=0.43，显著） | linalool.isovalerate（+，π^c=0.23，不显著） | X3.methylthio.propanol（-，π^c=0.04，不显著） |
| t2n | C | beta.phellandrene（-，π^c=0.19，不显著） | glycerol（-，π^c=0.16，不显著） | vinyl.caprylate（+，π^c=0.11，不显著） |
| orange | A | camphor（+，π^c=0.74，显著） | ethanol..v.v.（-，π^c=0.44，显著） | ethyl.furoate（-，π^c=0.09，不显著） |
| coriander | B | X2.methyl.benzofuran（+，π^c=0.07，显著） | cis.beta.terpineol（+，π^c=0.26，不显著） | hexyl.acetate（+，π^c=0.19，不显著） |
| clove | A | limonene（+，π^c=0.79，显著） | X2.ethyl.1.hexanol（-，π^c=0.16，不显著） | X2.acetylfuran（+，π^c=0.07，显著） |
| lactic | B | lactic_acid.mg.L.（+，π^c=0.35，显著） | isopentyl_acetate（-，π^c=0.53，显著） | phenethyl_alcohol（+，π^c=0.14，显著） |
| acetic | B | prenyl.isobutyrate（+，π^c=0.22，显著） | DES（-，π^c=0.32，显著） | ethyl_2.methyl_butyrate（+，π^c=0.07，显著） |
| alcohol | A | ethanol..v.v.（+，π^c=1.00，显著） | CO2.PSI.（+，π^c=0.67，显著） | bitterness（+，π^c=0.11，不显著） |
| aftertaste | B | ethanol..v.v.（+，π^c=0.52，显著） | X.E..2.nonenal（+，π^c=0.42，显著） | ethyl_isovalerate（+，π^c=0.56，显著） |
| body | A | ethanol..v.v.（+，π^c=1.00，显著） | color.EBC.（+，π^c=0.63，显著） | propylene.glycol（-，π^c=0.12，不显著） |
| co2 | A | ethyl_octanoate（+，π^c=0.85，显著） | CO2.PSI.（+，π^c=0.87，显著） | X3.methyl.2.hexanol（-，π^c=0.19，不显著） |
| overall | B | ethyl_acetate（+，π^c=0.56，显著） | X2.3.dihydro.farnesol（+，π^c=0.12，不显著） | X6.methyl.5.hepten.2.one（+，π^c=0.02，不显著） |

## 不显著或低稳定项（77 项）

- A_malt_all 第 2 位 X4.ethyl.phenol：显著，π^c=0.154（不稳定）
- A_malt_all 第 3 位 beta.phellandrene：显著，π^c=0.189（不稳定）
- A_malt_grain 第 2 位 X2.acetylfuran：不显著，π^c=0.095（不稳定）
- A_malt_grain 第 3 位 X.E..beta.ionone：不显著，π^c=0.056（不稳定）
- A_malt_bread 第 2 位 X4.ethyl.guaiacol：显著，π^c=0.162（不稳定）
- A_malt_bread 第 3 位 ethyl.tiglate：不显著，π^c=0.165（不稳定）
- A_malt_burn 第 2 位 DES：显著，π^c=0.167（不稳定）
- A_malt_burn 第 3 位 isopropyl.hexadecanoate：不显著，π^c=0.044（不稳定）
- A_hops_all 第 2 位 iron：显著，π^c=0.278（不稳定）
- A_hops_citrus 第 3 位 X6.methyl.5.hepten.2.one：显著，π^c=0.236（不稳定）
- A_hops_tropical 第 2 位 X.E..beta.ocimene：不显著，π^c=0.203（不稳定）
- A_hops_tropical 第 3 位 methyl.geranate：不显著，π^c=0.099（不稳定）
- A_hops_noble 第 3 位 X.E..2.nonenal：不显著，π^c=0.193（不稳定）
- A_hops_woody 第 1 位 trans.dihydro.alpha.terpinyl.acetate：显著，π^c=0.080（不稳定）
- A_hops_woody 第 2 位 X2Z.6Z.farnesol：不显著，π^c=0.187（不稳定）
- A_hops_woody 第 3 位 car.3.en.2.one：不显著，π^c=0.215（不稳定）
- A_esters_all 第 3 位 acetaldehyde：不显著，π^c=0.367（中等）
- A_esters_ethac 第 2 位 X2.methyl.1.butanol：显著，π^c=0.154（不稳定）
- A_esters_ethac 第 3 位 X2.3.dihydro.farnesol：不显著，π^c=0.091（不稳定）
- A_esters_isoaa 第 3 位 ethyl_decanoate：不显著，π^c=0.102（不稳定）
- A_esters_flower 第 1 位 beta.phellandrene：不显著，π^c=0.274（不稳定）
- A_esters_flower 第 2 位 n.hexanol：不显著，π^c=0.126（不稳定）
- A_esters_flower 第 3 位 acetic_acid.g.L.：不显著，π^c=0.077（不稳定）
- A_esters_fruity 第 2 位 X1.10.di.epi.cubenol：不显著，π^c=0.162（不稳定）
- A_esters_fruity 第 3 位 neryl.formate：显著，π^c=0.160（不稳定）
- F_malt_grain 第 1 位 isoborneol：显著，π^c=0.189（不稳定）
- F_malt_grain 第 2 位 X2.3.butanedione：不显著，π^c=0.166（不稳定）
- F_malt_grain 第 3 位 propyl.acetate：不显著，π^c=0.134（不稳定）
- F_malt_bread 第 1 位 acetaldehyde：显著，π^c=0.256（不稳定）
- F_malt_bread 第 3 位 neryl.formate：显著，π^c=0.185（不稳定）
- F_malt_cara 第 3 位 acetic_acid.g.L.：显著，π^c=0.221（不稳定）
- F_hops_all 第 3 位 linalool：显著，π^c=0.236（不稳定）
- F_hops_citrus 第 1 位 protein.g.L.：显著，π^c=0.255（不稳定）
- F_hops_citrus 第 2 位 nerol.oxide：显著，π^c=0.198（不稳定）
- F_hops_citrus 第 3 位 alpha.terpinyl.isobutanoate：不显著，π^c=0.086（不稳定）
- F_hops_tropical 第 1 位 geraniol：显著，π^c=0.255（不稳定）
- F_hops_tropical 第 2 位 DMS：显著，π^c=0.266（不稳定）
- F_hops_tropical 第 3 位 hexyl.acetate：不显著，π^c=0.345（中等）
- F_hops_noble 第 3 位 phenethyl_alcohol：不显著，π^c=0.118（不稳定）
- F_hops_woody 第 2 位 ethyl.hexadecanoate：显著，π^c=0.201（不稳定）
- F_hops_woody 第 3 位 isobutanol：不显著，π^c=0.271（不稳定）
- F_esters_all 第 3 位 pH：显著，π^c=0.197（不稳定）
- F_esters_ethac 第 2 位 E.nerolidol：显著，π^c=0.179（不稳定）
- F_esters_ethac 第 3 位 styrene：不显著，π^c=0.042（不稳定）
- F_esters_isoaa 第 3 位 tau.cadinol：不显著，π^c=0.096（不稳定）
- F_esters_flower 第 3 位 X.E..dihydro.apofarnesal：不显著，π^c=0.175（不稳定）
- acidity 第 2 位 isopentyl_acetate：显著，π^c=0.192（不稳定）
- acidity 第 3 位 eugenol：不显著，π^c=0.085（不稳定）
- bitternes 第 3 位 CO2.PSI.：显著，π^c=0.196（不稳定）
- sweetness 第 3 位 unfermentables：显著，π^c=0.306（不稳定）
- X4vg 第 3 位 betaglucan.mg.L.：显著，π^c=0.180（不稳定）
- diacetyl 第 2 位 furfural：显著，π^c=0.186（不稳定）
- diacetyl 第 3 位 phenethyl_acetate：不显著，π^c=0.050（不稳定）
- dms 第 2 位 terpinen.4.ol：不显著，π^c=0.287（不稳定）
- dms 第 3 位 unfermentables：不显著，π^c=0.211（不稳定）
- metallic 第 1 位 ethyl_isovalerate：显著，π^c=0.310（不稳定）
- metallic 第 3 位 ethyl_octanoate：显著，π^c=0.081（不稳定）
- stale_hops 第 2 位 linalool.isovalerate：不显著，π^c=0.226（不稳定）
- stale_hops 第 3 位 X3.methylthio.propanol：不显著，π^c=0.038（不稳定）
- t2n 第 1 位 beta.phellandrene：不显著，π^c=0.193（不稳定）
- t2n 第 2 位 glycerol：不显著，π^c=0.160（不稳定）
- t2n 第 3 位 vinyl.caprylate：不显著，π^c=0.112（不稳定）
- orange 第 3 位 ethyl.furoate：不显著，π^c=0.092（不稳定）
- coriander 第 1 位 X2.methyl.benzofuran：显著，π^c=0.069（不稳定）
- coriander 第 2 位 cis.beta.terpineol：不显著，π^c=0.255（不稳定）
- coriander 第 3 位 hexyl.acetate：不显著，π^c=0.193（不稳定）
- clove 第 2 位 X2.ethyl.1.hexanol：不显著，π^c=0.162（不稳定）
- clove 第 3 位 X2.acetylfuran：显著，π^c=0.072（不稳定）
- lactic 第 3 位 phenethyl_alcohol：显著，π^c=0.139（不稳定）
- acetic 第 1 位 prenyl.isobutyrate：显著，π^c=0.224（不稳定）
- acetic 第 2 位 DES：显著，π^c=0.316（不稳定）
- acetic 第 3 位 ethyl_2.methyl_butyrate：显著，π^c=0.071（不稳定）
- alcohol 第 3 位 bitterness：不显著，π^c=0.111（不稳定）
- body 第 3 位 propylene.glycol：不显著，π^c=0.117（不稳定）
- co2 第 3 位 X3.methyl.2.hexanol：不显著，π^c=0.193（不稳定）
- overall 第 2 位 X2.3.dihydro.farnesol：不显著，π^c=0.125（不稳定）
- overall 第 3 位 X6.methyl.5.hepten.2.one：不显著，π^c=0.023（不稳定）

## 主方法与 M3 Top-1 不同（24 维）

- A_malt_bread：主方法 color.EBC.，M3 X2.acetylpyrrole
- A_malt_burn：主方法 color.EBC.，M3 isoamyl_alcohol
- A_hops_citrus：主方法 citronellol，M3 CO2.PSI.
- A_hops_tropical：主方法 DMS，M3 n.hexanol
- A_hops_woody：主方法 trans.dihydro.alpha.terpinyl.acetate，M3 X.E..ortho.methoxy.cinnamaldehyde
- A_esters_ethac：主方法 ethyl_acetate，M3 phenethyl_alcohol
- A_esters_flower：主方法 beta.phellandrene，M3 n.hexanol
- F_malt_all：主方法 ethanol..v.v.，M3 color.EBC.
- F_malt_bread：主方法 acetaldehyde，M3 X2.acetylpyrrole
- F_malt_burn：主方法 ethanol..v.v.，M3 color.EBC.
- F_hops_tropical：主方法 geraniol，M3 n.hexanol
- F_hops_noble：主方法 bitterness，M3 ethyl_octanoate
- F_hops_woody：主方法 bitterness，M3 car.3.en.2.one
- F_esters_isoaa：主方法 n.hexanol，M3 unfermentables
- F_esters_fruity：主方法 DES，M3 GFS
- sweetness：主方法 ethyl_decanoate，M3 GFS
- dms：主方法 protein.g.L.，M3 unfermentables
- metallic：主方法 ethyl_isovalerate，M3 iron
- stale_hops：主方法 X2.undecanol，M3 iron
- t2n：主方法 beta.phellandrene，M3 H2S
- coriander：主方法 X2.methyl.benzofuran，M3 ethyl..E..linalool
- acetic：主方法 prenyl.isobutyrate，M3 lactic_acid.mg.L.
- aftertaste：主方法 ethanol..v.v.，M3 bitterness
- co2：主方法 ethyl_octanoate，M3 CO2.PSI.

## 共变标记（80 项）

- A_malt_all 第 1 位 color.EBC.：≥5 维
- A_malt_all 第 2 位 X4.ethyl.phenol：风格内 CI 含 0
- A_malt_all 第 3 位 beta.phellandrene：≥5 维
- A_malt_grain 第 2 位 X2.acetylfuran：风格内 CI 含 0
- A_malt_grain 第 3 位 X.E..beta.ionone：风格内 CI 含 0
- A_malt_bread 第 1 位 color.EBC.：≥5 维
- A_malt_bread 第 3 位 ethyl.tiglate：风格内 CI 含 0
- A_malt_cara 第 1 位 color.EBC.：≥5 维
- A_malt_burn 第 1 位 color.EBC.：≥5 维
- A_malt_burn 第 2 位 DES：风格内 CI 含 0
- A_hops_all 第 1 位 ethyl_octanoate：≥5 维
- A_hops_tropical 第 3 位 methyl.geranate：风格内 CI 含 0
- A_hops_noble 第 1 位 ethyl_octanoate：≥5 维
- A_hops_woody 第 3 位 car.3.en.2.one：风格内 CI 含 0
- A_esters_all 第 1 位 n.hexanol：≥5 维
- A_esters_all 第 3 位 acetaldehyde：全部啤酒 OAV<1
- A_esters_isoaa 第 1 位 n.hexanol：≥5 维
- A_esters_isoaa 第 3 位 ethyl_decanoate：风格内 CI 含 0；全部啤酒 OAV<1
- A_esters_flower 第 1 位 beta.phellandrene：≥5 维
- A_esters_flower 第 2 位 n.hexanol：≥5 维
- A_esters_flower 第 3 位 acetic_acid.g.L.：风格内 CI 含 0
- A_esters_fruity 第 1 位 beta.phellandrene：风格内 CI 含 0；≥5 维
- A_esters_fruity 第 3 位 neryl.formate：风格内 CI 含 0
- F_malt_all 第 1 位 ethanol..v.v.：风格内 CI 含 0；≥5 维
- F_malt_all 第 2 位 color.EBC.：≥5 维
- F_malt_grain 第 3 位 propyl.acetate：风格内 CI 含 0
- F_malt_bread 第 1 位 acetaldehyde：全部啤酒 OAV<1
- F_malt_bread 第 2 位 X2.acetylpyrrole：风格内 CI 含 0
- F_malt_cara 第 1 位 color.EBC.：≥5 维
- F_malt_cara 第 2 位 acetaldehyde：风格内 CI 含 0；全部啤酒 OAV<1
- F_malt_cara 第 3 位 acetic_acid.g.L.：风格内 CI 含 0
- F_malt_burn 第 1 位 ethanol..v.v.：≥5 维
- F_malt_burn 第 2 位 pH：风格内 CI 含 0
- F_malt_burn 第 3 位 color.EBC.：风格内 CI 含 0；≥5 维
- F_hops_all 第 1 位 bitterness：≥5 维
- F_hops_all 第 2 位 ethyl_octanoate：≥5 维
- F_hops_citrus 第 1 位 protein.g.L.：风格内 CI 含 0；≥5 维
- F_hops_citrus 第 3 位 alpha.terpinyl.isobutanoate：风格内 CI 含 0
- F_hops_tropical 第 2 位 DMS：风格内 CI 含 0
- F_hops_noble 第 1 位 bitterness：≥5 维
- F_hops_noble 第 2 位 ethyl_octanoate：≥5 维
- F_hops_noble 第 3 位 phenethyl_alcohol：风格内 CI 含 0
- F_hops_woody 第 1 位 bitterness：≥5 维
- F_hops_woody 第 3 位 isobutanol：风格内 CI 含 0；全部啤酒 OAV<1
- F_esters_all 第 1 位 protein.g.L.：≥5 维
- F_esters_all 第 2 位 n.hexanol：≥5 维
- F_esters_ethac 第 3 位 styrene：风格内 CI 含 0
- F_esters_isoaa 第 1 位 n.hexanol：≥5 维
- F_esters_isoaa 第 2 位 protein.g.L.：≥5 维
- F_esters_flower 第 1 位 beta.phellandrene：≥5 维
- F_esters_fruity 第 2 位 protein.g.L.：≥5 维
- F_esters_fruity 第 3 位 beta.phellandrene：风格内 CI 含 0；≥5 维
- acidity 第 1 位 pH：风格内 CI 含 0
- acidity 第 2 位 isopentyl_acetate：风格内 CI 含 0
- bitternes 第 1 位 bitterness：≥5 维
- bitternes 第 2 位 ethyl_octanoate：≥5 维
- sweetness 第 1 位 ethyl_decanoate：全部啤酒 OAV<1
- sweetness 第 2 位 protein.g.L.：≥5 维
- X4vg 第 3 位 betaglucan.mg.L.：风格内 CI 含 0
- diacetyl 第 2 位 furfural：全部啤酒 OAV<1
- diacetyl 第 3 位 phenethyl_acetate：全部啤酒 OAV<1
- dms 第 1 位 protein.g.L.：≥5 维
- metallic 第 1 位 ethyl_isovalerate：风格内 CI 含 0
- metallic 第 3 位 ethyl_octanoate：风格内 CI 含 0；≥5 维
- stale_hops 第 3 位 X3.methylthio.propanol：风格内 CI 含 0
- t2n 第 1 位 beta.phellandrene：≥5 维
- t2n 第 2 位 glycerol：风格内 CI 含 0
- orange 第 2 位 ethanol..v.v.：风格内 CI 含 0；≥5 维
- orange 第 3 位 ethyl.furoate：风格内 CI 含 0
- clove 第 2 位 X2.ethyl.1.hexanol：风格内 CI 含 0
- clove 第 3 位 X2.acetylfuran：风格内 CI 含 0
- lactic 第 1 位 lactic_acid.mg.L.：风格内 CI 含 0
- acetic 第 1 位 prenyl.isobutyrate：风格内 CI 含 0
- acetic 第 3 位 ethyl_2.methyl_butyrate：风格内 CI 含 0
- alcohol 第 1 位 ethanol..v.v.：≥5 维
- alcohol 第 3 位 bitterness：风格内 CI 含 0；≥5 维
- aftertaste 第 1 位 ethanol..v.v.：≥5 维
- body 第 1 位 ethanol..v.v.：≥5 维
- body 第 2 位 color.EBC.：≥5 维
- co2 第 1 位 ethyl_octanoate：≥5 维

## 阈值证据（有阈值且已校准的 Top-3 化合物 26 个，OAV≥1 啤酒比例）

- protein.g.L.：1.00
- linalool：0.98
- ethanol..v.v.：0.98
- citronellol：0.92
- bitterness：0.86
- isopentyl_acetate：0.84
- geraniol：0.76
- DMS：0.67
- ethyl_acetate：0.66
- acetic_acid.g.L.：0.62
- iron：0.48
- X2.3.butanedione：0.46
- ethyl_octanoate：0.26
- DES：0.18
- GFS：0.18
- lactic_acid.mg.L.：0.17
- ethyl_2.methyl_butyrate：0.07
- X4VG：0.06
- glycerol：0.03
- phenethyl_alcohol：0.02
- ethyl_isovalerate：0.01
- acetaldehyde：0.00
- ethyl_decanoate：0.00
- isobutanol：0.00
- furfural：0.00
- phenethyl_acetate：0.00

## 同模块成员与模块可分检验

- A_malt_all 第 1 位 color.EBC.：protein.g.L.
- A_malt_all 第 2 位 X4.ethyl.phenol：neryl.formate|X4.ethyl.guaiacol
- A_malt_all 第 3 位 beta.phellandrene：terpinolene
- A_malt_grain 第 1 位 terpinen.4.ol：isoborneol|ethyl..E..linalool
- A_malt_grain 第 2 位 X2.acetylfuran：X2.acetylpyrrole
- A_malt_grain 第 3 位 X.E..beta.ionone：X1.octen.3.ol
- A_malt_bread 第 1 位 color.EBC.：protein.g.L.
- A_malt_bread 第 2 位 X4.ethyl.guaiacol：X4.ethyl.phenol|neryl.formate
- A_malt_cara 第 1 位 color.EBC.：protein.g.L.
- A_malt_burn 第 1 位 color.EBC.：protein.g.L.
- A_malt_burn 第 3 位 isopropyl.hexadecanoate：manool.oxide
- A_hops_all 第 1 位 ethyl_octanoate：ethyl_decanoate
- A_hops_all 第 3 位 linalool：X1.alpha.terpineol
- A_hops_tropical 第 3 位 methyl.geranate：X2.undecanol|caryophyllene.oxide|humulene.epoxide.II|alpha.muurolol|tau.muurolol
- A_hops_noble 第 1 位 ethyl_octanoate：ethyl_decanoate
- A_hops_noble 第 3 位 X.E..2.nonenal：cis.dihydro.beta.terpineol
- A_hops_woody 第 1 位 trans.dihydro.alpha.terpinyl.acetate：isopentyl.hexanoate|X2.butyloctanoate|X2.phenyl.ethyl.butyrate|X.Z..beta.farnesene|pentyl.octanoate|X2.methylbutyl.octanoate|X1.2.dihydro.8.hydroxy..2E..linalool|ethyl.tetradecanoate|ethyl.pentadecanoate|ethyl.hexadecanoate
- A_hops_woody 第 3 位 car.3.en.2.one：dimethoxy..E..citral
- A_esters_all 第 1 位 n.hexanol：isobutyl_acetate|isopentyl_acetate
- A_esters_ethac 第 1 位 ethyl_acetate：ethyl_hexanoate|glycerol|ethanol..v.v.
- A_esters_ethac 第 2 位 X2.methyl.1.butanol：X1.propanol|n.pentanol|isopentyl.formate
- A_esters_ethac 第 3 位 X2.3.dihydro.farnesol：E.nerolidol|dihydroisocaryophyllene.epoxide|X2Z.6E.farnesol
- A_esters_isoaa 第 1 位 n.hexanol：isobutyl_acetate|isopentyl_acetate
- A_esters_isoaa 第 3 位 ethyl_decanoate：ethyl_octanoate
- A_esters_flower 第 1 位 beta.phellandrene：terpinolene
- A_esters_flower 第 2 位 n.hexanol：isobutyl_acetate|isopentyl_acetate
- A_esters_fruity 第 1 位 beta.phellandrene：terpinolene
- A_esters_fruity 第 2 位 X1.10.di.epi.cubenol：neryl.propanoate|alpha.calacorene|alloaromadendrene.epoxide|caryophylla.4.12..8.13..dien.5.alpha.ol
- A_esters_fruity 第 3 位 neryl.formate：X4.ethyl.phenol|X4.ethyl.guaiacol
- F_malt_all 第 1 位 ethanol..v.v.：ethyl_acetate|ethyl_hexanoate|glycerol
- F_malt_all 第 2 位 color.EBC.：protein.g.L.
- F_malt_grain 第 1 位 isoborneol：terpinen.4.ol|ethyl..E..linalool
- F_malt_grain 第 2 位 X2.3.butanedione：ethyl.pentanoate
- F_malt_bread 第 2 位 X2.acetylpyrrole：X2.acetylfuran
- F_malt_bread 第 3 位 neryl.formate：X4.ethyl.phenol|X4.ethyl.guaiacol
- F_malt_cara 第 1 位 color.EBC.：protein.g.L.
- F_malt_burn 第 1 位 ethanol..v.v.：ethyl_acetate|ethyl_hexanoate|glycerol
- F_malt_burn 第 3 位 color.EBC.：protein.g.L.
- F_hops_all 第 2 位 ethyl_octanoate：ethyl_decanoate
- F_hops_all 第 3 位 linalool：X1.alpha.terpineol
- F_hops_citrus 第 1 位 protein.g.L.：color.EBC.
- F_hops_citrus 第 2 位 nerol.oxide：ethyl.hex..2E..enoate|isobutyl.hexanoate
- F_hops_noble 第 2 位 ethyl_octanoate：ethyl_decanoate
- F_hops_woody 第 2 位 ethyl.hexadecanoate：isopentyl.hexanoate|trans.dihydro.alpha.terpinyl.acetate|X2.butyloctanoate|X2.phenyl.ethyl.butyrate|X.Z..beta.farnesene|pentyl.octanoate|X2.methylbutyl.octanoate|X1.2.dihydro.8.hydroxy..2E..linalool|ethyl.tetradecanoate|ethyl.pentadecanoate
- F_hops_woody 第 3 位 isobutanol：isoamyl_alcohol
- F_esters_all 第 1 位 protein.g.L.：color.EBC.
- F_esters_all 第 2 位 n.hexanol：isobutyl_acetate|isopentyl_acetate
- F_esters_ethac 第 1 位 ethyl_acetate：ethyl_hexanoate|glycerol|ethanol..v.v.
- F_esters_ethac 第 2 位 E.nerolidol：dihydroisocaryophyllene.epoxide|X2.3.dihydro.farnesol|X2Z.6E.farnesol
- F_esters_isoaa 第 1 位 n.hexanol：isobutyl_acetate|isopentyl_acetate
- F_esters_isoaa 第 2 位 protein.g.L.：color.EBC.
- F_esters_flower 第 1 位 beta.phellandrene：terpinolene
- F_esters_fruity 第 2 位 protein.g.L.：color.EBC.
- F_esters_fruity 第 3 位 beta.phellandrene：terpinolene
- acidity 第 2 位 isopentyl_acetate：isobutyl_acetate|n.hexanol
- bitternes 第 2 位 ethyl_octanoate：ethyl_decanoate
- sweetness 第 1 位 ethyl_decanoate：ethyl_octanoate
- sweetness 第 2 位 protein.g.L.：color.EBC.
- X4vg 第 2 位 neryl.formate：X4.ethyl.phenol|X4.ethyl.guaiacol
- dms 第 1 位 protein.g.L.：color.EBC.
- dms 第 2 位 terpinen.4.ol：isoborneol|ethyl..E..linalool
- metallic 第 1 位 ethyl_isovalerate：ethyl_2.methyl_butyrate
- metallic 第 3 位 ethyl_octanoate：ethyl_decanoate
- stale_hops 第 1 位 X2.undecanol：methyl.geranate|caryophyllene.oxide|humulene.epoxide.II|alpha.muurolol|tau.muurolol
- t2n 第 1 位 beta.phellandrene：terpinolene
- t2n 第 2 位 glycerol：ethyl_acetate|ethyl_hexanoate|ethanol..v.v.
- orange 第 2 位 ethanol..v.v.：ethyl_acetate|ethyl_hexanoate|glycerol
- orange 第 3 位 ethyl.furoate：X5.methylfurfural
- clove 第 3 位 X2.acetylfuran：X2.acetylpyrrole
- lactic 第 2 位 isopentyl_acetate：isobutyl_acetate|n.hexanol
- acetic 第 1 位 prenyl.isobutyrate：ethyl.benzoate|ethyl.phenylacetate|perillyl.acetate
- acetic 第 3 位 ethyl_2.methyl_butyrate：ethyl_isovalerate
- alcohol 第 1 位 ethanol..v.v.：ethyl_acetate|ethyl_hexanoate|glycerol
- aftertaste 第 1 位 ethanol..v.v.：ethyl_acetate|ethyl_hexanoate|glycerol
- aftertaste 第 2 位 X.E..2.nonenal：cis.dihydro.beta.terpineol
- aftertaste 第 3 位 ethyl_isovalerate：ethyl_2.methyl_butyrate
- body 第 1 位 ethanol..v.v.：ethyl_acetate|ethyl_hexanoate|glycerol
- body 第 2 位 color.EBC.：protein.g.L.
- co2 第 1 位 ethyl_octanoate：ethyl_decanoate
- co2 第 3 位 X3.methyl.2.hexanol：X3.furanmethanol|ortho.cresol|n.octanol
- overall 第 1 位 ethyl_acetate：ethyl_hexanoate|glycerol|ethanol..v.v.
- overall 第 2 位 X2.3.dihydro.farnesol：E.nerolidol|dihydroisocaryophyllene.epoxide|X2Z.6E.farnesol

## 标注

- stale_hops 行 `note` 列：同 barnyard；barnyard 不单独建模。

## 未解释项

- 全部 147 项的化学解释待 Modeling 撰写（Coding 不生成机理解释）。

## 敏感性汇总

数值见 `code/data/processed/q2/sensitivity/sensitivity_summary.csv`。

```
                   arm  n_dims  n_permutation  n_bootstrap  mean_jaccard_compound  mean_jaccard_module  n_top1_changed  n_grade_changed  pi_c_pos1  pi_c_pos2  pi_c_pos3  n_stable  n_medium  n_unstable  n_A  n_B  n_C
              baseline      49         1000.0       1000.0                  1.000                1.000             0.0              0.0      0.566      0.357      0.189      29.0      43.0        75.0 22.0 25.0  2.0
               tau_0.5      49         1000.0       1000.0                  1.000                1.000             0.0              1.0      0.564      0.358      0.191      28.0      44.0        75.0 21.0 26.0  2.0
               tau_0.7      49         1000.0       1000.0                  0.990                0.993             0.0              0.0      0.567      0.357      0.191      29.0      43.0        75.0 22.0 25.0  2.0
  cand_plus_composites      49         1000.0       1000.0                  0.863                0.867             5.0              1.0      0.537      0.336      0.198      27.0      40.0        80.0 21.0 26.0  2.0
cand_plus_internal_std      49         1000.0       1000.0                  0.984                0.984             0.0              0.0      0.566      0.356      0.188      29.0      43.0        75.0 22.0 25.0  2.0
             na_lowest      49         1000.0       1000.0                  0.980                0.980             0.0              0.0      0.565      0.355      0.182      29.0      42.0        76.0 22.0 25.0  2.0
  attachment8_conflict      49         1000.0       1000.0                  0.943                0.943             1.0              0.0      0.571      0.364      0.180      29.0      40.0        78.0 22.0 25.0  2.0
      boot_fixed_graph      49         1000.0       1000.0                  1.000                1.000             0.0              0.0      0.566      0.357      0.189      29.0      43.0        75.0 22.0 25.0  2.0
        boot_beer_unit      49         1000.0       1000.0                  1.000                1.000             0.0              2.0      0.596      0.374      0.199      29.0      45.0        73.0 22.0 25.0  2.0
      sensory_anchored      49         1000.0       1000.0                  0.692                0.696             6.0              2.0      0.570      0.321      0.190      28.0      37.0        82.0 22.0 25.0  2.0
     pairwise_deletion      49            NaN          NaN                    NaN                  NaN             NaN              NaN        NaN        NaN        NaN       NaN       NaN         NaN  NaN  NaN  NaN
```

- tau_0.5 等级改变：A_esters_all:A->B
- cand_plus_composites 等级改变：A_esters_all:A->B
- boot_beer_unit 等级改变：A_esters_all:A->B|F_malt_burn:B->A
- sensory_anchored 等级改变：F_malt_all:B->A|diacetyl:A->B
- pairwise_deletion：不适用：逐对删除各系数样本集合不同，不能形成统一相关矩阵，不定义完整选择流程；附件 6 44 行复算最大误差 6.66e-16；有效样本数范围 223–250；与列均值边际系数最大差 0.075
