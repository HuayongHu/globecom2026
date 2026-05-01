
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

METHOD_LABELS={'rag_cra':'RAG-CRA','rag_cra_no_api':'RAG-CRA w/o LLM','ml_policy':'ML policy','pso':'PSO','ga':'GA','de':'DE','random':'Random','direct_llm':'Direct LLM','rag_only':'RAG only','rag_cra_no_refine':'w/o refine','rag_cra_no_robust':'w/o robust'}
def label(x): return METHOD_LABELS.get(str(x),str(x))
def setup_style():
    plt.rcParams.update({'font.size':10,'axes.labelsize':11,'axes.titlesize':12,'legend.fontsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'figure.dpi':140,'savefig.dpi':600,'pdf.fonttype':42,'ps.fonttype':42,'axes.grid':True,'grid.alpha':0.22,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,path,save_pdf=True):
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(Path(path).with_suffix('.png'), bbox_inches='tight')
    if save_pdf:
        fig.savefig(Path(path).with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
def load_results(out_dir):
    df=pd.read_csv(Path(out_dir)/'all_results.csv')
    if 'status' in df.columns: df=df[df['status'].astype(str).str.lower().isin(['ok','nan']) | df['status'].isna()].copy()
    return df
def summary(df,suite):
    d=df[df['suite']==suite]
    if d.empty: return pd.DataFrame()
    agg={'robust':('robust_score','mean'),'robust_std':('robust_score','std'),'cvar_pd':('cvar_pd','mean'),'worst_pd':('worst_pd','mean'),'viol':('risk_violation_rate','mean'),'evals':('eval_count','mean'),'time':('runtime_sec','mean'),'mem':('peak_memory_mb','mean'),'n':('robust_score','count')}
    if 'flop_proxy' in d.columns: agg['flops']=('flop_proxy','mean')
    g=d.groupby('method').agg(**agg).reset_index(); g['ci95']=1.96*g['robust_std'].fillna(0)/np.sqrt(np.maximum(g['n'],1)); return g.sort_values('robust',ascending=False)
def paired_deltas(df,suite,base='rag_cra'):
    p=df[df['suite']==suite].pivot_table(index=['scenario_id','seed'],columns='method',values='robust_score',aggfunc='mean')
    if base not in p.columns: return pd.DataFrame()
    rows=[]
    for m in p.columns:
        if m==base: continue
        pair=p[[base,m]].dropna(); delta=pair[base]-pair[m]
        if delta.empty: continue
        rows.append({'suite':suite,'baseline':m,'n_pairs':len(delta),'mean_delta':delta.mean(),'median_delta':delta.median(),'ci95_delta':1.96*delta.std(ddof=1)/np.sqrt(max(len(delta),1)) if len(delta)>1 else 0,'win_rate':(delta>0).mean()})
    return pd.DataFrame(rows).sort_values('mean_delta')
def plot_paired_delta(df,out_dir,suite,save_pdf=True):
    d=paired_deltas(df,suite); 
    if d.empty: return d
    fig,ax=plt.subplots(figsize=(7.2,max(3.2,0.45*len(d)+1.3))); y=np.arange(len(d))
    ax.barh(y,d['mean_delta'],xerr=d['ci95_delta'],capsize=3); ax.axvline(0,color='black',lw=1)
    ax.set_yticks(y); ax.set_yticklabels([f'RAG-CRA - {label(m)}' for m in d['baseline']]); ax.set_xlabel('Paired robust-score difference'); ax.set_title(f'{suite}: paired deltas')
    for yy,(_,r) in zip(y,d.iterrows()): ax.text(r['mean_delta'],yy+0.15,f"win={r['win_rate']:.2f}",fontsize=8,ha='center')
    save(fig,Path(out_dir)/f'fig8_paired_delta_{suite}',save_pdf); return d
def plot_ood_risk(df,out_dir,save_pdf=True):
    s=summary(df,'ood_stress'); keep=[m for m in ['rag_cra','pso','ga','de','rag_cra_no_api','random','ml_policy'] if m in set(s['method'])]; s=s.set_index('method').loc[keep].reset_index()
    fig,axs=plt.subplots(2,2,figsize=(8.4,6.2)); metrics=[('robust','Robust score'),('cvar_pd','CVaR Pd'),('worst_pd','Worst Pd'),('viol','Violation rate (lower better)')]
    for ax,(col,title) in zip(axs.ravel(),metrics):
        x=np.arange(len(s)); ax.bar(x,s[col]); ax.set_xticks(x); ax.set_xticklabels([label(m) for m in s['method']],rotation=25,ha='right'); ax.set_title(title); ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.suptitle('OOD risk certificate metrics',y=1.02); save(fig,Path(out_dir)/'fig9_ood_risk_certificate_metrics',save_pdf)
def plot_efficiency(df,out_dir,suite,save_pdf=True):
    s=summary(df,suite)
    fig,ax=plt.subplots(figsize=(6.8,4.4))
    ax.scatter(s['evals'],s['robust'],s=120,alpha=.82)
    for _,r in s.iterrows():
        ax.annotate(label(r['method']),(r['evals'],r['robust']),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Mean simulator calls per scenario')
    ax.set_ylabel('Mean robust score')
    ax.set_title(f'{suite}: simulator-call efficiency')
    save(fig,Path(out_dir)/f'fig10_efficiency_{suite}',save_pdf)
def plot_flop(df,out_dir,suite,save_pdf=True):
    s=summary(df,suite)
    if 'flops' not in s.columns:
        return
    fig,ax1=plt.subplots(figsize=(7.0,4.4))
    s=s.sort_values('robust',ascending=False)
    x=np.arange(len(s))
    ax1.bar(x,s['robust'],alpha=.75,label='Robust score')
    ax1.set_ylabel('Mean robust score')
    ax1.set_xticks(x); ax1.set_xticklabels([label(m) for m in s['method']],rotation=25,ha='right')
    ax2=ax1.twinx()
    ax2.plot(x,s['flops']/1e6,marker='o',linestyle='--',label='FLOP proxy')
    ax2.set_ylabel('FLOP/token proxy (1e6)')
    ax1.set_title(f'{suite}: robust score and FLOP/token proxy')
    save(fig,Path(out_dir)/f'fig11_flop_tradeoff_{suite}',save_pdf)
def plot_ablation_waterfall(df,out_dir,save_pdf=True):
    s=summary(df,'ablation').set_index('method'); order=[m for m in ['direct_llm','rag_only','rag_cra_no_robust','rag_cra_no_refine','rag_cra_no_api','rag_cra'] if m in s.index]; vals=s.loc[order,'robust']; base=vals.iloc[0]; delta=vals-base
    fig,ax=plt.subplots(figsize=(7.2,4.4)); x=np.arange(len(order)); ax.bar(x,delta.values); ax.axhline(0,color='black',lw=1); ax.set_xticks(x); ax.set_xticklabels([label(m) for m in order],rotation=25,ha='right'); ax.set_ylabel('Robust-score gain over Direct LLM'); ax.set_title('Ablation gain decomposition')
    for xi,v in zip(x,delta.values): ax.text(xi,v+0.005,f'{v:+.3f}',ha='center',fontsize=8)
    save(fig,Path(out_dir)/'fig12_ablation_gain_waterfall',save_pdf)
def plot_diff_to_best(df,out_dir,save_pdf=True):
    rows=[]
    for suite in ['main_nominal','ood_stress']:
        p=df[df['suite']==suite].pivot_table(index='family',columns='method',values='robust_score',aggfunc='mean')
        if 'rag_cra' not in p: continue
        for fam,val in (p['rag_cra']-p.max(axis=1)).items(): rows.append({'suite':suite,'family':fam,'delta_to_best':val})
    dd=pd.DataFrame(rows); 
    if dd.empty: return
    mat=dd.pivot(index='family',columns='suite',values='delta_to_best').fillna(0); fig,ax=plt.subplots(figsize=(6.6,max(3.5,.38*len(mat)+1.6))); im=ax.imshow(mat.values,aspect='auto',vmin=min(-.12,mat.values.min()),vmax=0)
    ax.set_xticks(np.arange(mat.shape[1])); ax.set_xticklabels(['Main' if c=='main_nominal' else 'OOD' for c in mat.columns]); ax.set_yticks(np.arange(mat.shape[0])); ax.set_yticklabels(mat.index); ax.set_title('RAG-CRA difference to best method')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]): ax.text(j,i,f'{mat.values[i,j]:+.3f}',ha='center',va='center',fontsize=8)
    cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.02); cb.set_label('RAG-CRA - best robust score'); save(fig,Path(out_dir)/'fig13_difference_to_best_heatmap',save_pdf); dd.to_csv(Path(out_dir)/'v41_difference_to_best_by_family.csv',index=False)
def plot_llm_usage(df,out_dir,save_pdf=True):
    if 'prompt_tokens' not in df.columns:
        return
    d=df[df['method'].isin(['rag_cra','direct_llm','rag_cra_no_refine','rag_cra_no_robust'])].copy()
    if d.empty:
        return
    d['tokens']=d.get('prompt_tokens',0).fillna(0)+d.get('completion_tokens',0).fillna(0)
    s=d.groupby(['suite','method']).agg(tokens=('tokens','mean'),parsed=('parsed_candidate_count','mean'),robust=('robust_score','mean')).reset_index()
    s['name']=s['suite'].astype(str)+' / '+s['method'].map(label)
    fig,ax1=plt.subplots(figsize=(8.2,4.8))
    x=np.arange(len(s))
    ax1.bar(x,s['robust'],alpha=.75,label='Robust score')
    ax1.set_ylabel('Mean robust score')
    ax1.set_xticks(x); ax1.set_xticklabels(s['name'],rotation=35,ha='right')
    ax2=ax1.twinx(); ax2.plot(x,s['tokens'],marker='o',linestyle='--',label='Tokens')
    ax2.set_ylabel('Mean LLM tokens')
    ax1.set_title('LLM usage versus robust performance')
    save(fig,Path(out_dir)/'fig14_llm_token_usage',save_pdf)
    s.to_csv(Path(out_dir)/'v41_llm_usage_summary.csv',index=False)
def write_notes(df,out_dir):
    lines=['# v4.1 Paper-Readiness Notes','']
    for suite in ['main_nominal','ood_stress','ablation']:
        s=summary(df,suite); 
        if s.empty: continue
        best=s.iloc[0]; lines += [f'## {suite}',f"Best robust mean: **{label(best['method'])}** ({best['robust']:.4f})."]
        if 'rag_cra' in set(s['method']):
            r=s[s['method']=='rag_cra'].iloc[0]; lines.append(f"RAG-CRA robust={r['robust']:.4f}, CVaR Pd={r['cvar_pd']:.4f}, worst Pd={r['worst_pd']:.4f}, violation={r['viol']:.4f}, evals={r['evals']:.1f}.")
        lines.append('')
    lines += ['## Claim boundary','- Safe: verifier-guided, sample-risk-aware, simulator-call efficient versus classical optimizers.','- Unsafe: LLM API alone is the dominant performance source; v4 shows only small gains over clean no-API in aggregate.','- Caution: PSO remains stronger in several OOD CVaR/worst-Pd metrics.']
    Path(out_dir,'v41_paper_readiness_notes.md').write_text('\n'.join(lines),encoding='utf-8')
def make_v41_figures(out_dir,save_pdf=True,semantic_dir=None):
    setup_style()
    out_dir=Path(out_dir)
    df=load_results(out_dir)
    ds=[]
    for suite in ['main_nominal','ood_stress','ablation']:
        print(f'[v4.1] paired delta: {suite}', flush=True)
        d=plot_paired_delta(df,out_dir,suite,save_pdf)
        if not d.empty:
            ds.append(d)
    if ds:
        pd.concat(ds,ignore_index=True).to_csv(out_dir/'v41_paired_deltas.csv',index=False)
    print('[v4.1] ood risk metrics', flush=True)
    plot_ood_risk(df,out_dir,save_pdf)
    print('[v4.1] efficiency main', flush=True)
    plot_efficiency(df,out_dir,'main_nominal',save_pdf)
    print('[v4.1] efficiency ood', flush=True)
    plot_efficiency(df,out_dir,'ood_stress',save_pdf)
    print('[v4.1] flop main', flush=True)
    plot_flop(df,out_dir,'main_nominal',save_pdf)
    print('[v4.1] flop ood', flush=True)
    plot_flop(df,out_dir,'ood_stress',save_pdf)
    print('[v4.1] ablation waterfall', flush=True)
    plot_ablation_waterfall(df,out_dir,save_pdf)
    print('[v4.1] difference to best', flush=True)
    plot_diff_to_best(df,out_dir,save_pdf)
    print('[v4.1] llm usage', flush=True)
    plot_llm_usage(df,out_dir,save_pdf)
    print('[v4.1] notes', flush=True)
    write_notes(df,out_dir)
    if semantic_dir and Path(semantic_dir).exists():
        try:
            from .semantic_stress import plot_semantic_results
            plot_semantic_results(semantic_dir,save_pdf=save_pdf)
        except Exception as e:
            (out_dir/'v41_semantic_plot_error.txt').write_text(repr(e),encoding='utf-8')
    print('[v4.1] done', flush=True)
