"""Build human-readable dictionaries and remaining read-only evidence."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import openpyxl
import pandas as pd
from dataAudit.audit_data import save_json, write_csv, read_csv_raw
from dataAudit.audit_deep import numeric, sheet_frame

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'experimentalCode'/'results'/'data_audit'; RAW=ROOT/'code'/'data'/'raw'


def main():
    chem=pd.read_csv(OUT/'chemical_name_mapping.csv',keep_default_na=False)
    sens=pd.read_csv(OUT/'sensory_name_mapping.csv',keep_default_na=False)
    transform=pd.read_csv(OUT/'transform_verification.csv',keep_default_na=False)
    profiles=pd.read_csv(OUT/'variable_dictionary.csv',keep_default_na=False)
    w=openpyxl.load_workbook(RAW/'8.xlsx',data_only=True)
    sn=sheet_frame(w,'Sensory names'); cal=sheet_frame(w,'Calibrated chemical dataset')
    c1=read_csv_raw(RAW/'1.csv')[0]; c7=read_csv_raw(RAW/'7.csv')[0]
    cnum=numeric(cal.iloc[:,3:]); a=numeric(c1)
    dc=pd.read_csv(OUT/'calibrated_discordances.csv')
    special={}; extra=[]
    stats={}
    for i,row in chem.iterrows():
        name=row.csv1_field; p=profiles[(profiles.file=='1.csv')&(profiles.column==name)].iloc[0].to_dict()
        if i<37:mode='log10 已由附件8正值配对验证 非正值存在半最小正值替代'
        elif name.endswith('_sum'):mode='综合指标 公式与变换过程未提供'
        elif name in set(transform.field):mode='线性尺度 与附件8多数配对一致 存在需核实差异'
        else:mode='疑似对数信号 无独立变换前数据 底数待核实'
        r={**row.to_dict(),'role':'综合指标' if name.endswith('_sum') or name=='kcalperc' else '化学或理化特征',
           'transform_interpretation':mode,**{k:p[k] for k in ('missing','missing_rate','min','median','max','negative','zero','examples')}}
        extra.append(r); special[('1.csv',name)]=r
    write_csv(OUT/'chemical_dictionary.csv',extra)
    template=pd.read_csv(OUT/'template_field_mapping.csv',keep_default_na=False)
    aliases={'A_esters_fruity':'A_ester_fruity','F_esters_fruity':'F_ester_fruity','X4vg':'4vg'}
    extra=[]
    for _,row in sens.iterrows():
        name=row.csv4_field; p=profiles[(profiles.file=='4.csv')&(profiles.column==name)].iloc[0].to_dict()
        tm=template[template.template_field==aliases.get(name,name)].iloc[0]
        r={**row.to_dict(),'template_field':tm.template_field,'template_cell_beer1':tm.formula_cell,
           'template_formula':tm.formula,'template_labels':tm.labels,
           'scale_interpretation':'中心化或标准化后聚合值的具体过程待核实 无原始0至6还原参数',
           **{k:p[k] for k in ('missing','missing_rate','min','max','mean','std_ddof1')}}
        extra.append(r);special[('4.csv',name)]=r
    write_csv(OUT/'sensory_dictionary.csv',extra)
    rb=sn[sn.dataset=='Ratebeer']; extra=[]
    for name in c7.columns[3:]:
        p=profiles[(profiles.file=='7.csv')&(profiles.column==name)].iloc[0].to_dict()
        meta=rb[rb.R_name==name]
        r={'field':name,'channel':name.split('_',1)[0],'descriptor':name.split('_',1)[1],
           'metadata_match_rows':len(meta),'provided_long_names':' | '.join(meta.CleanName_PaperLong.astype(str)),
           'source':'论坛描述符 附件8指向Ratebeer','scale_interpretation':'连续衍生数值 具体计数和变换过程未提供',
           **{k:p[k] for k in ('missing','missing_rate','min','max','mean','std_ddof1')}}
        extra.append(r);special[('7.csv',name)]=r
    write_csv(OUT/'forum_dictionary.csv',extra)
    enriched=[]
    for _,p in profiles.iterrows():
        r=p.to_dict(); file=r['file'];name=r['column'];s=special.get((file,name),{})
        if name=='beer':role='样本编号 唯一整数1至250'
        elif name=='beer_id':role='样本标识 按原始文本保留'
        elif name=='tasting_category_fine':role='风格标签 22类'
        elif file in ('2.csv','5.csv','6.csv'):
            role='矩阵行标签' if int(r['position'])==1 else 'Spearman相关系数列 非样本测量值'
        elif file=='1.csv':role=s.get('role','化学特征')
        elif file=='4.csv':role='训练品评小组感官属性'
        else:role='论坛感官描述符衍生值'
        r.update(role=role,unit=s.get('unit',''),long_name=s.get('chemical_name',s.get('CleanName_PaperLong',s.get('provided_long_names',''))),
                 interpretation=s.get('transform_interpretation',s.get('scale_interpretation','')))
        enriched.append(r)
    write_csv(OUT/'variable_dictionary.csv',enriched)
    # Largest positive values are evidence records, not automatic exclusions.
    extremes=[]
    for name in c1.columns[3:]:
        v=a[name]; idx=int(v.idxmax()); med=v.median()
        meta=chem[chem.csv1_field==name].iloc[0]
        extremes.append({'field':name,'beer_at_max':c1.iloc[idx].beer,'csv_row':idx+2,'max_literal':c1.iloc[idx][name],
                         'median':float(med),'max_over_median':float(v.max()/med) if med>0 else None,
                         'unit_from_metadata':meta.unit})
    write_csv(OUT/'extreme_values_review.csv',extremes)
    negative_cal=[{'beer':int(cal.iloc[i].beer),'field':cnum.columns[j],'calibrated_value':float(cnum.iat[i,j])}
                  for i,j in zip(*np.where(cnum.to_numpy()<0))]
    forum_fields=set(c7.columns[3:])
    stats['forum_metadata']={'ratebeer_rows':len(rb),'unique_R_names':rb.R_name.nunique(),
                             'fields_in_7_with_mapping':len(set(rb.R_name)&forum_fields),
                             'metadata_fields_not_in_7':sorted(set(rb.R_name)-forum_fields),
                             'fields_in_7_without_mapping':len(forum_fields-set(rb.R_name)),
                             'duplicate_R_names':rb[rb.R_name.duplicated(keep=False)].R_name.unique().tolist()}
    stats['calibrated_discordances']={'cells':len(dc),'fields':dc.field.nunique(),'beers':dc.beer.nunique(),
                                     'by_field':dc.groupby('field').size().to_dict()}
    stats['negative_calibrated']=negative_cal
    stats['common_signal_floor']=float(10**(-1.81781854))
    stats['original_brief_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('2026*.docx')}
    stats['draft_sha256']=hashlib.sha256((ROOT/'赛题与草稿.txt').read_bytes()).hexdigest()
    # Verify full source ID vectors also against both workbook sample datasets.
    idref=dict(zip(c1.beer,c1.beer_id))
    stats['workbook_id_checks']={}
    for name in ('Calibrated chemical dataset','Beer Prices'):
        d=sheet_frame(w,name)
        stats['workbook_id_checks'][name]={'unique_beer':int(d.beer.nunique()),
                                          'unique_beer_id':int(d.beer_id.nunique()),
                                          'matches_csv1':sum(str(int(row.beer_id))==idref[str(int(row.beer))] for _,row in d.iterrows())}
    save_json(OUT/'supplement_findings.json',stats)
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
