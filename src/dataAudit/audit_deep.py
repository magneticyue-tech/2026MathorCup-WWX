"""Cross-attachment diagnostics. Does not clean, fill, merge or overwrite source data.

Run after audit_data.py. Alignment is a temporary diagnostic using unique beer.
Candidate identifier and name mappings are evidence tables, not processed data.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from dataAudit.audit_data import read_csv_raw, save_json, write_csv, scalar, MISSING

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "data_audit"


def sheet_frame(wb, name):
    rows = list(wb[name].values)
    return pd.DataFrame(rows[1:], columns=rows[0])


def numeric(d):
    return d.apply(lambda s: pd.to_numeric(s.mask(s.astype(str).str.strip().isin(MISSING)), errors="coerce"))


def correlations(d, method):
    # pandas' compiled matrix implementation supports pairwise ranks and ties.
    return d.corr(method=method).to_numpy()


def compare_arrays(a, b, mask=None):
    ok = np.isfinite(a) & np.isfinite(b)
    if mask is not None: ok &= mask
    delta = np.abs(a[ok] - b[ok])
    return {"n":len(delta), "mae":float(delta.mean()) if len(delta) else None,
            "max_abs":float(delta.max()) if len(delta) else None,
            "within_1e-8":int((delta < 1e-8).sum()), "within_1e-6":int((delta < 1e-6).sum())}


def main():
    ds={k:read_csv_raw(RAW/f"{k}.csv")[0] for k in ('1','2','4','5','6','7')}
    ns={k:numeric(v) for k,v in ds.items()}
    w=openpyxl.load_workbook(RAW/'8.xlsx',data_only=True)
    th=sheet_frame(w,'Sensory thresholds'); sn=sheet_frame(w,'Sensory names')
    cal=sheet_frame(w,'Calibrated chemical dataset'); shap=sheet_frame(w,'GBM SHAP Values'); price=sheet_frame(w,'Beer Prices')
    x=ns['1'].iloc[:,3:].copy(); y=ns['4'].iloc[:,3:].copy(); chemical_cols=list(x.columns)
    result={}

    # Unique sample keys and candidate restoration of reduced-precision IDs.
    identity={}; candidates=[]
    for k in ('1','4','7'):
        d=ds[k]
        identity[k]={"beer_unique":d.beer.nunique(), "beer_min":int(ns[k].beer.min()), "beer_max":int(ns[k].beer.max()),
                     "id_unique":d.beer_id.nunique(), "duplicate_id_rows":int(d.beer_id.duplicated(keep=False).sum()),
                     "id_length_counts":dict(Counter(d.beer_id.str.len())),
                     "id_scientific_rows":int(d.beer_id.str.contains('[Ee]').sum()),
                     "style_counts":dict(Counter(d.tasting_category_fine))}
    ref=ds['1'].set_index('beer')
    for k in ('4','7'):
        d=ds[k].set_index('beer')
        identity[k]['beer_sets_equal_to_1']=set(d.index)==set(ref.index)
        identity[k]['beer_order_equal_to_1']=list(d.index)==list(ref.index)
        identity[k]['style_matches_by_beer']=int((d.loc[ref.index,'tasting_category_fine']==ref.tasting_category_fine).sum())
        rounded=[format(Decimal(v),'.5E') for v in ref.beer_id]
        observed=[format(Decimal(v),'.5E') for v in d.loc[ref.index,'beer_id']]
        identity[k]['six_significant_digits_matches']=sum(a==b for a,b in zip(rounded,observed))
        # The export reduced IDs with 12+ digits; shorter integer IDs survived.
        # Compare exact integers for the latter, not six-significant rounding.
        export_keys=[str(Decimal(format(Decimal(v),'.5E'))) if len(v)>=12 else str(Decimal(v)) for v in ref.beer_id]
        target_keys=[str(Decimal(v)) for v in d.loc[ref.index,'beer_id']]
        groups={}
        for beer, rid in zip(ref.index,export_keys):groups.setdefault(Decimal(rid),[]).append(beer)
        identity[k]['rounded_key_unique_rows']=sum(len(groups.get(Decimal(v),[]))==1 for v in target_keys)
        identity[k]['rounded_key_ambiguous_rows']=sum(len(groups.get(Decimal(v),[]))>1 for v in target_keys)
        identity[k]['rounded_key_max_group']=max(map(len,groups.values()))
        for beer in ref.index:
            v=format(Decimal(d.loc[beer,'beer_id']),'.5E')
            candidates.append({'table':k,'beer':beer,'source_beer_id':d.loc[beer,'beer_id'],
                               'candidate_beer_id_from_1':ref.loc[beer,'beer_id'],'rounded_matches':v==format(Decimal(ref.loc[beer,'beer_id']),'.5E'),
                               'rounding_only_candidate_count':len(groups.get(Decimal(d.loc[beer,'beer_id']),[])),
                               'style_matches':d.loc[beer,'tasting_category_fine']==ref.loc[beer,'tasting_category_fine']})
    result['identity']=identity; write_csv(OUT/'identifier_candidates.csv',candidates)

    # Exact metadata maps; positional mappings are kept explicitly provisional.
    thgroups={str(k):g for k,g in th.groupby('field_name_R')}
    chemical_map=[]
    for c, label in zip(chemical_cols,ds['2'].columns[1:]):
        g=thgroups.get(c,pd.DataFrame())
        meta=g.iloc[0].to_dict() if len(g)==1 else {}
        chemical_map.append({'csv1_field':c,'csv2_label':label,'metadata_rows':len(g),**meta,
                             'matrix_label_exact_short':label==meta.get('chemical_name_short'),
                             'matrix_label_exact_long':label==meta.get('chemical_name')})
    write_csv(OUT/'chemical_name_mapping.csv',chemical_map)
    panel=sn[sn.dataset=='Trained panel'].set_index('R_name')
    sensory_map=[]
    # CSV5 swaps the order of co2, aftertaste and body relative to CSV4.
    csv5_fields=list(y.columns[:46])+['co2','aftertaste','body','overall']
    for c,label in zip(csv5_fields,ds['5'].columns[1:]):
        meta=panel.loc[c].to_dict()
        sensory_map.append({'csv4_field':c,'csv5_label':label,**meta})
    write_csv(OUT/'sensory_name_mapping.csv',sensory_map)
    result['metadata']={'chemical_exact_R_matches':sum(r['metadata_rows']==1 for r in chemical_map),
                        'chemical_matrix_shortname_matches':sum(r['matrix_label_exact_short'] for r in chemical_map),
                        'chemical_matrix_longname_matches':sum(r['matrix_label_exact_long'] for r in chemical_map),
                        'chemical_unmatched_or_duplicate':[r['csv1_field'] for r in chemical_map if r['metadata_rows']!=1],
                        'sensory_meta_panel_rows':len(panel), 'panel_type_counts':dict(Counter(panel.DataType)),
                        'sensory_meta_ratebeer_rows':int((sn.dataset=='Ratebeer').sum()),
                        'chemical_metadata_unused':th[~th.field_name_R.isin(chemical_cols)].field_name_R.tolist()}

    # Validate base against independent, positive calibrated concentrations.
    cal=cal.set_index('beer'); cal.index=cal.index.astype(str)
    cal=cal.loc[ds['1'].beer]; calnum=numeric(cal)
    transform=[]; calibrated_discordances=[]; calibrated_missing=[]
    for c in cal.columns[2:]:
        xv=x[c].to_numpy(float); cv=calnum[c].to_numpy(float)
        ok=np.isfinite(xv)&np.isfinite(cv)&(cv>0)
        e10=compare_arrays(xv[ok],np.log10(cv[ok])); eln=compare_arrays(xv[ok],np.log(cv[ok])); eid=compare_arrays(xv[ok],cv[ok])
        zero=(cv<=0)&np.isfinite(xv)
        positive=cv[np.isfinite(cv)&(cv>0)]
        zvals=np.unique(xv[zero]); floor=float(10**zvals[0]) if len(zvals)==1 else None
        is_log=chemical_cols.index(c)<37
        predicted=np.full(cv.shape,np.nan)
        if is_log: predicted[cv>0]=np.log10(cv[cv>0])
        else: predicted=cv.copy()
        matched=0; discord=0
        for i,original in enumerate(ds['1'][c]):
            if np.isfinite(xv[i]) and np.isfinite(predicted[i]):
                # A CSV literal represents a rounded interval of half its last
                # printed decimal place, including scientific-notation cells.
                quantum=Decimal(1).scaleb(Decimal(original).as_tuple().exponent)
                tol=float(abs(quantum))/2 + 1e-11
                delta=float(abs(xv[i]-predicted[i]))
                if delta<=tol:matched+=1
                else:
                    discord+=1
                    calibrated_discordances.append({'beer':ds['1'].iloc[i].beer,'field':c,'csv1_literal':original,
                                                    'calibrated_8':cv[i],'expected_mode':'log10' if is_log else 'identity',
                                                    'absolute_difference':delta,'csv_rounding_tolerance':tol})
            if not np.isfinite(cv[i]) or not np.isfinite(xv[i]):
                calibrated_missing.append({'beer':ds['1'].iloc[i].beer,'field':c,'csv1_literal':original,
                                           'calibrated_8':scalar(cv[i]),'csv1_missing':bool(not np.isfinite(xv[i])),
                                           'calibrated_missing':bool(not np.isfinite(cv[i])),'calibrated_zero':bool(cv[i]==0)})
        r={'field':c,'positive_pairs':int(ok.sum()),'log10_max_abs':e10['max_abs'],'log10_mae':e10['mae'],
           'ln_mae':eln['mae'],'identity_mae':eid['mae'],'zeros_in_calibrated':int((cv==0).sum()),
           'negatives_in_calibrated':int((cv<0).sum()),'mode_for_rounding_check':'log10' if is_log else 'identity',
           'matches_within_printed_rounding':matched,'material_discordances':discord,
           'nonpositive_log_distinct':len(zvals),'nonpositive_log_values':json.dumps(zvals.tolist()),
           'implied_floor':floor,'min_positive_calibrated':float(positive.min()) if len(positive) else None,
           'floor_over_min_positive':float(floor/positive.min()) if floor is not None and len(positive) else None,
           'missing_csv1':int(np.isnan(xv).sum()),'missing_calibrated':int(np.isnan(cv).sum()),
           'missing_masks_equal':bool(np.array_equal(np.isnan(xv),np.isnan(cv)))}
        transform.append(r)
    write_csv(OUT/'transform_verification.csv',transform)
    write_csv(OUT/'calibrated_discordances.csv',calibrated_discordances)
    write_csv(OUT/'calibrated_missing_comparison.csv',calibrated_missing)
    result['transform']={'fields':len(transform),'positive_pairs':sum(r['positive_pairs'] for r in transform),
                         'log10_fields_max_error_lt_1e-7':sum(r['log10_max_abs']<1e-7 for r in transform),
                         'max_log10_error':max(r['log10_max_abs'] for r in transform),
                         'zero_fields':sum(r['zeros_in_calibrated']>0 for r in transform),
                         'zero_cells':sum(r['zeros_in_calibrated'] for r in transform),
                         'zero_single_floor_fields':sum(r['zeros_in_calibrated']>0 and r['nonpositive_log_distinct']==1 for r in transform),
                         'zero_halfmin_fields':sum(r['zeros_in_calibrated']>0 and r['floor_over_min_positive'] is not None and abs(r['floor_over_min_positive']-.5)<1e-5 for r in transform),
                         'missing_mask_differences':[r for r in transform if not r['missing_masks_equal']]}
    result['transform']['material_discordances_after_csv_rounding']=len(calibrated_discordances)
    result['transform']['log37']={
        'positive_pairs':sum(r['positive_pairs'] for r in transform[:37]),
        'max_error':max(r['log10_max_abs'] for r in transform[:37]),
        'nonpositive_cells':sum(r['zeros_in_calibrated']+r['negatives_in_calibrated'] for r in transform[:37]),
        'negative_cells':sum(r['negatives_in_calibrated'] for r in transform[:37]),
        'missing_calibrated':sum(r['missing_calibrated'] for r in transform[:37]),
        'halfmin_nonpositive_fields':sum(r['nonpositive_log_distinct']==1 and r['floor_over_min_positive'] is not None and abs(r['floor_over_min_positive']-.5)<1e-5 for r in transform[:37])}
    result['transform']['linear33']={
        'finite_pairs':sum(r['matches_within_printed_rounding']+r['material_discordances'] for r in transform[37:]),
        'matches_after_rounding':sum(r['matches_within_printed_rounding'] for r in transform[37:]),
        'material_discordances':sum(r['material_discordances'] for r in transform[37:]),
        'missing_calibrated':sum(r['missing_calibrated'] for r in transform[37:]),
        'csv1_missing_on_calibrated_zero':sum(z['csv1_missing'] and z['calibrated_zero'] for z in calibrated_missing)}
    remaining=[c for c in chemical_cols if c not in cal.columns and not c.endswith('_sum')]
    uncal=x[remaining]
    negvals=uncal.to_numpy()[uncal.to_numpy()<0]
    result['transform']['uncalibrated157']={'fields':len(remaining),'negative_values':int(len(negvals)),
                                          'negative_distinct_values':np.unique(negvals).tolist(),
                                          'no_independent_pretransform_reference':True}

    # Matrix diagnostics and reproduction, including every off-diagonal pair.
    matrix_stats={}
    for k in ('2','5','6'):
        vals=ns[k].iloc[:,1:].to_numpy(float)
        square=vals.shape[0]==vals.shape[1]
        matrix_stats[k]={'shape':list(vals.shape),'min':float(vals.min()),'max':float(vals.max()),
                         'outside_range':int((np.abs(vals)>1+1e-12).sum()),'missing':int(np.isnan(vals).sum()),
                         'row_labels_unique':bool(ds[k].iloc[:,0].is_unique),'column_labels_unique':bool(ds[k].columns[1:].is_unique)}
        if square:
            eig=np.linalg.eigvalsh(vals)
            matrix_stats[k].update(symmetry_max_abs=float(np.abs(vals-vals.T).max()),
                                    diag_max_abs_from_one=float(np.abs(np.diag(vals)-1).max()),
                                    row_labels_equal_column_labels=ds[k].iloc[:,0].tolist()==ds[k].columns[1:].tolist(),
                                    min_eigenvalue=float(eig.min()),negative_eigenvalues_lt_minus_1e_8=int((eig < -1e-8).sum()))
    pair_evidence=[]
    for k,data,fields in [('2',x,chemical_cols),('5',y[csv5_fields],csv5_fields)]:
        target=ns[k].iloc[:,1:].to_numpy(float); mask=np.triu(np.ones(target.shape,bool),1)
        sp=correlations(data,'spearman'); pe=correlations(data,'pearson')
        matrix_stats[k]['spearman_pairwise']=compare_arrays(target,sp,mask)
        matrix_stats[k]['pearson_pairwise']=compare_arrays(target,pe,mask)
        complete=data.notna().all(axis=1)
        matrix_stats[k]['listwise_n']=int(complete.sum())
        matrix_stats[k]['spearman_listwise']=compare_arrays(target,correlations(data[complete],'spearman'),mask)
        for i,j in zip(*np.where(mask)):
            pair_evidence.append({'table':k,'row_field':fields[i],'col_field':fields[j],
                                  'given':target[i,j],'spearman':scalar(sp[i,j]),'pearson':scalar(pe[i,j]),
                                  'spearman_abs_error':scalar(abs(target[i,j]-sp[i,j])),
                                  'pair_n':int((data.iloc[:,i].notna()&data.iloc[:,j].notna()).sum()),
                                  'either_field_missing':bool(data.iloc[:,i].isna().any() or data.iloc[:,j].isna().any())})
    label_to_field=dict(zip(ds['2'].columns[1:],chemical_cols))
    crossfields=[label_to_field[c] for c in ds['6'].iloc[:,0]]
    xy=pd.concat([x[crossfields].reset_index(drop=True), y[csv5_fields].reset_index(drop=True)],axis=1)
    target=ns['6'].iloc[:,1:].to_numpy(float)
    for method in ('spearman','pearson'):
        pred=correlations(xy,method)[:len(crossfields),len(crossfields):]
        matrix_stats['6'][method+'_pairwise']=compare_arrays(target,pred)
        if method=='spearman':sp=pred
        else:pe=pred
    for i,c in enumerate(crossfields):
        for j,s in enumerate(csv5_fields):
            pair_evidence.append({'table':'6','row_field':c,'col_field':s,'given':target[i,j],
                                  'spearman':scalar(sp[i,j]),'pearson':scalar(pe[i,j]),
                                  'spearman_abs_error':scalar(abs(target[i,j]-sp[i,j])),
                                  'pair_n':int((x[c].notna()&y[s].notna()).sum()),'either_field_missing':bool(x[c].isna().any())})
    result['matrices']=matrix_stats
    write_csv(OUT/'correlation_reproduction.csv',pair_evidence)

    # Workbook 8 block semantics and missingness. Blank cells in layout sheets
    # are not counted as unmeasured observations.
    workbook8={}; workbook8_profiles=[]
    for name in w.sheetnames:
        if name=='Sheet2':
            workbook8[name]={'empty':True};continue
        d=sheet_frame(w,name); miss=d.apply(lambda s:s.map(lambda v: pd.isna(v) or (isinstance(v,str) and v.strip() in MISSING)))
        workbook8[name]={'rows':len(d),'columns':len(d.columns),'missing_cells':int(miss.to_numpy().sum()),
                          'missing_columns':{str(c):int(miss[c].sum()) for c in d.columns if miss[c].any()}}
        for c in d.columns:
            workbook8_profiles.append({'sheet':name,'column':str(c),'rows':len(d),'missing':int(miss[c].sum()),
                                       'missing_rate':float(miss[c].mean()),'unique_nonmissing':int(d.loc[~miss[c],c].nunique())})
    conc=shap[shap.data=='concentration'].copy(); sh=shap[shap.data=='SHAP'].copy()
    conc.index=conc.beer.astype(int).astype(str); conc=conc.loc[ds['1'].beer]
    cf=numeric(conc[chemical_cols]); cf.index=x.index
    result['shap']={'block_counts':dict(Counter(shap.data)),
                    'concentration_equals_csv1_including_missing':bool(np.allclose(cf,x,atol=1e-12,rtol=0,equal_nan=True)),
                    'concentration_diff':compare_arrays(cf.to_numpy(float),x.to_numpy(float)),
                    'concentration_missing':int(cf.isna().to_numpy().sum()),
                    'base_prediction_by_block':{str(k):[scalar(v) for v in g.base_prediction.unique()] for k,g in shap.groupby('data')},
                    'mdi_sum':float(pd.to_numeric(sheet_frame(w,'GBM importance MDI').iloc[:,0]).sum())}
    result['workbook8']=workbook8;write_csv(OUT/'8_columns.csv',workbook8_profiles)

    # Missingness patterns and source-scale diagnostics; no deletion rules.
    z=ns['7'].iloc[:,3:]
    result['scales']={}
    for name,d in [('4',y),('7',z)]:
        good=d.loc[:,d.notna().any()]
        result['scales'][name]={'numeric_fields':len(d.columns),'all_missing_fields':d.columns[d.isna().all()].tolist(),
                                'range':[float(good.min().min()),float(good.max().max())],
                                'means_range':[float(good.mean().min()),float(good.mean().max())],
                                'std_ddof1_range':[float(good.std().min()),float(good.std().max())],
                                'negative_cells':int((good<0).to_numpy().sum()),'noninteger_cells':int((~np.isclose(good,np.round(good),atol=1e-8,rtol=0)).sum()),
                                'all_columns_mean0_std1_at_1e-6':bool((good.mean().abs()<1e-6).all() and ((good.std()-1).abs()<1e-6).all())}
    miss=x.isna(); styles=ds['1'].tasting_category_fine
    missing_patterns=Counter(tuple(miss.columns[row.to_numpy()]) for _,row in miss.iterrows())
    result['missing1']={'feature_cells':int(x.size),'cells':int(miss.to_numpy().sum()),
                        'rows':int(miss.any(axis=1).sum()),'row_max':int(miss.sum(axis=1).max()),
                        'patterns':[{'fields':list(p),'rows':n} for p,n in missing_patterns.most_common()],
                        'by_style':[{"style":s,'n':int((styles==s).sum()),'missing_rows':int(miss[styles==s].any(axis=1).sum()),
                                     'missing_cells':int(miss[styles==s].to_numpy().sum())} for s in styles.unique()]}
    missstyle=[]
    for c in miss.columns[miss.any()]:
        for s in styles.unique():
            ix=styles==s; missstyle.append({'field':c,'style':s,'samples':int(ix.sum()),'missing':int(miss.loc[ix,c].sum()),'rate':float(miss.loc[ix,c].mean())})
    write_csv(OUT/'missing_by_style.csv',missstyle)
    floors=[]
    for c in chemical_cols:
        v=x[c].dropna(); cnt=v.value_counts(); mv=float(v.min())
        floors.append({'field':c,'n':len(v),'min':mv,'min_count':int((v==mv).sum()),'min_fraction':float((v==mv).mean()),
                       'unique_values':v.nunique(),'most_common_value':float(cnt.index[0]),'most_common_count':int(cnt.iloc[0])})
    write_csv(OUT/'floors_and_ties.csv',floors)

    # Template formulas, source-field links, and empty input evidence.
    form=openpyxl.load_workbook(RAW/'3.xlsx',data_only=False)
    cached=openpyxl.load_workbook(RAW/'3.xlsx',data_only=True)
    summary=form['summary']; template_map=[]
    for j in range(2,summary.max_column+1):
        header=summary.cell(1,j).value; formula=summary.cell(2,j).value
        coord=formula.split('!')[-1].replace('$','')
        cell=form['beer_1'][coord]
        txt=[str(form['beer_1'].cell(r,col).value) for r in (cell.row-1,cell.row) for col in range(max(1,cell.column-15),cell.column)
             if form['beer_1'].cell(r,col).data_type!='f' and isinstance(form['beer_1'].cell(r,col).value,str)]
        template_map.append({'template_field':header,'summary_cell':summary.cell(1,j).coordinate,
                             'formula_cell':coord,'formula':cell.value,'labels':' | '.join(txt),
                             'cached_template_value':cached['beer_1'][coord].value})
    write_csv(OUT/'template_field_mapping.csv',template_map)
    nonblank_inputs=[]
    for ws in list(form)[:12]:
        # Rows above printed 0-6 scales contain response boxes; subtype C/S boxes.
        for row in ws:
            for cell in row:
                if cell.data_type=='f' and cell.column in (17,33):
                    for refcell in re.findall(r'ISBLANK\(([A-Z]+\d+)\)',cell.value):
                        v=ws[refcell].value
                        if v is not None: nonblank_inputs.append({'sheet':ws.title,'cell':refcell,'value':v})
    result['template']={'sheets':len(form.sheetnames),'beer_sheets':12,'summary_fields_excluding_beer':summary.max_column-1,
                        'response_boxes_nonblank':nonblank_inputs,'all_formula_count':sum(c.data_type=='f' for ws in form for row in ws for c in row),
                        'template_fields_missing_from_4':[r['template_field'] for r in template_map if r['template_field'] not in y.columns],
                        'csv4_fields_missing_from_template':[c for c in y if c not in [r['template_field'] for r in template_map]]}
    # Input immutability is checked against the initial manifest.
    initial=json.loads((OUT/'raw_manifest.json').read_text(encoding='utf-8'))
    result['raw_unchanged']=all(hashlib.sha256((RAW/r['file']).read_bytes()).hexdigest()==r['sha256'] for r in initial)
    save_json(OUT/'deep_findings.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('missing1','workbook8','identity')},ensure_ascii=False,indent=2))
    print('Identity',json.dumps(identity,ensure_ascii=False))
    print('Workbook8',json.dumps(workbook8,ensure_ascii=False))


if __name__=='__main__':main()
