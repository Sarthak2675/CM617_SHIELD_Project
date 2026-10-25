"""Reproduce all reported numerical experiments and editable CSV/LaTeX tables."""
from pathlib import Path
from dataclasses import replace,asdict
import json,time,platform,hashlib
import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shield import *

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'; FIG=ROOT/'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
COLORS={EV.name:'#d46a31',SC.name:'#237e92'}
AUDIT=[]

def execute(p,c,label):
    df,diag=run(p,c)
    AUDIT.append(dict(run_id=len(AUDIT)+1,run=label,**asdict(c),strategy=p.name,
                      **{'plant_'+k:v for k,v in asdict(p).items()},**diag))
    return df

def metrics(df,bare,shade):
    times=df.time_h.to_numpy()
    if len(times)<2:
        raise ValueError('At least two samples are required to infer the interval.')
    dt_h=float(times[1]-times[0])
    if not np.allclose(np.diff(times),dt_h,rtol=0,atol=1e-10):
        raise ValueError('Metrics require uniformly spaced simulation records.')
    for control in (bare,shade):
        if len(control)!=len(df) or not np.allclose(control.time_h,times,rtol=0,atol=1e-10):
            raise ValueError('Counterfactual time axes must match.')
    mask=(df.hour>=10)&(df.hour<18)
    delta=bare.T2_C.to_numpy()-df.T2_C.to_numpy()
    extra=shade.T2_C.to_numpy()-df.T2_C.to_numpy()
    df=df.copy(); df['cooling_C']=delta;df['extra_cooling_C']=extra
    end=(df.day==df.day.max())&(df.hour>=14)&(df.hour<16)
    m=dict(cooling_degree_h=float(delta[mask].sum()*dt_h),transp_cooling_degree_h=float(extra[mask].sum()*dt_h),lastday_cooling_C=float(delta[end].mean()),transp_mm=float(df.transp_mm.sum()),irrigation_mm=float(df.irrigation_mm.sum()),final_psi_MPa=float(psi_from_theta(df.theta_end.iloc[-1])))
    return df,m

def savefig(name):
    plt.savefig(FIG/(name+'.pdf'));plt.savefig(FIG/(name+'.png'),dpi=220);plt.close()

def main():
    start=time.time()
    # One-day fixed-potential baseline; compare traits at identical water status.
    rows=[]; traces=[]
    base=Config(days=1,fixed_psi=-.3,policy='none')
    bare=execute(EV,replace(base,mode='bare'),'static_bare')
    for lai in [1.,3.,5.]:
        shade=execute(EV,replace(base,LAI=lai,mode='shade'),f'static_shade_{lai}')
        for p in [EV,SC]:
            for psi in [-.3,-1.8]:
                c=replace(base,LAI=lai,fixed_psi=psi)
                d=execute(p,c,f'static_{p.name}_{lai}_{psi}')
                # fixed-window mean avoids timestep/nearest-grid timing ambiguity.
                pick=(d.hour>=14)&(d.hour<16)
                cooling=float((bare.T2_C-d.T2_C)[pick].mean())
                ex=float((shade.T2_C-d.T2_C)[pick].mean())
                rows.append(dict(strategy=p.name,LAI=lai,psi_MPa=psi,cooling_C=cooling,transp_increment_C=ex,transp_mm=d.transp_mm.sum()))
                if lai==3 and psi==-.3:
                    d['scenario']=p.name;traces.append(d)
        if lai==3:
            shade['scenario']='shade';traces.append(shade)
    bare['scenario']='bare';traces.append(bare)
    static=pd.DataFrame(rows);static.to_csv(OUT/'static_factorial.csv',index=False)
    pd.concat(traces).to_csv(OUT/'static_timeseries.csv',index=False)
    resilience=[]
    for (name,lai),g in static.groupby(['strategy','LAI']):
        wet=g[g.psi_MPa==-.3].iloc[0]; dry=g[g.psi_MPa==-1.8].iloc[0]
        resilience.append(dict(strategy=name,LAI=lai,total_CRI=dry.cooling_C/wet.cooling_C,transp_CRI=dry.transp_increment_C/wet.transp_increment_C))
    pd.DataFrame(resilience).to_csv(OUT/'resilience.csv',index=False)
    print('Static factorial complete',flush=True)
    # Seven-day finite-water factorial + timing comparison.
    base=Config()
    bare=execute(EV,replace(base,mode='bare',policy='none'),'week_bare')
    summaries=[]; alltr=[]
    for lai in [1.,3.,5.]:
        shade=execute(EV,replace(base,LAI=lai,mode='shade',policy='none'),f'week_shade_{lai}')
        for p in [EV,SC]:
            for budget in [0.,6.,12.,24.]:
                policies=['none'] if budget==0 else ['daily']
                if lai==3 and budget==12:policies+=['front','threshold']
                for policy in policies:
                    c=replace(base,LAI=lai,budget=budget,policy=policy)
                    label=f'{p.name}_L{lai}_B{budget}_{policy}'
                    df=execute(p,c,label)
                    df,m=metrics(df,bare,shade)
                    summaries.append(dict(strategy=p.name,LAI=lai,budget_mm=budget,policy=policy,**m))
                    df['strategy']=p.name;df['LAI']=lai;df['budget_mm']=budget;df['policy']=policy
                    alltr.append(df)
    summary=pd.DataFrame(summaries); summary.to_csv(OUT/'dynamic_summary.csv',index=False)
    times=pd.concat(alltr,ignore_index=True);times.to_csv(OUT/'dynamic_timeseries.csv',index=False)
    # Incremental benefit of actual irrigation relative to the same no-irrigation canopy.
    gains=[]
    for _,r in summary.iterrows():
        zero=summary[(summary.strategy==r.strategy)&(summary.LAI==r.LAI)&(summary.budget_mm==0)].iloc[0]
        gains.append((r.cooling_degree_h-zero.cooling_degree_h)/r.irrigation_mm if r.irrigation_mm>0 else np.nan)
    summary['incremental_degree_h_per_mm']=gains
    summary.to_csv(OUT/'dynamic_summary.csv',index=False)
    print('Dynamic factorial complete',flush=True)
    # 36 explicitly selected sensitivity scenarios, paired traits. Not a posterior.
    sens=[]
    for depth in [.3,.6,1.2]:
        for tau in [75.,150.,300.]:
            for psi in [-.10,-.30]:
                for heat in [0.,3.]:
                    cc=replace(base,root_depth=depth,tau=tau,psi_initial=psi,heat=heat)
                    bb=execute(EV,replace(cc,mode='bare',policy='none'),'sensitivity_bare')
                    ss=execute(EV,replace(cc,mode='shade',policy='none'),'sensitivity_shade')
                    for p in [EV,SC]:
                        dd=execute(p,cc,'sensitivity_full')
                        _,m=metrics(dd,bb,ss)
                        sens.append(dict(root_depth_m=depth,tau_s=tau,initial_psi_MPa=psi,heat_increment_C=heat,strategy=p.name,**m))
    sensitivity=pd.DataFrame(sens);sensitivity.to_csv(OUT/'sensitivity.csv',index=False)
    print('Sensitivity complete',flush=True)
    # Separate correlated archetype traits: 2 x 2 Kp/P50 ablation.
    trait_rows=[]
    for kp in [.0055,.008]:
        for p50 in [-1.4,-3.]:
            pp=replace(EV,Kp=kp,psi50=p50)
            dd=execute(pp,base,'independent_traits')
            _,mm=metrics(dd,bare,execute(EV,replace(base,mode='shade',policy='none'),'trait_shade'))
            trait_rows.append(dict(Kp=kp,psi50_MPa=p50,**mm))
    pd.DataFrame(trait_rows).to_csv(OUT/'independent_traits.csv',index=False)
    # Explicitly audit the artificial lower storage cutoff and drawdown domain.
    cutoff_rows=[]
    for floor in [-4.,-6.,-8.]:
        for p in [EV,SC]:
            cc=replace(base,psi_floor=floor)
            dd=execute(p,cc,'cutoff_audit')
            _,mm=metrics(dd,bare,execute(EV,replace(cc,mode='shade',policy='none'),'cutoff_shade'))
            cutoff_rows.append(dict(psi_floor_MPa=floor,strategy=p.name,**mm))
    pd.DataFrame(cutoff_rows).to_csv(OUT/'cutoff_audit.csv',index=False)

    # Numerical tests on dynamic LAI=3, daily irrigation; same window and integral.
    conv=[]
    for dt,n,nd,nchi in [(600,12,81,61),(300,12,81,61),(150,12,81,61),(300,24,81,61),(150,24,81,61),(300,12,161,121),(300,12,41,31)]:
        cc=replace(base,dt=dt,N=n,nd=nd,nchi=nchi)
        bb=execute(EV,replace(cc,mode='bare',policy='none'),'convergence_bare')
        for p in [EV,SC]:
            dd=execute(p,cc,'convergence_full')
            mask=(dd.hour>=10)&(dd.hour<18); end=(dd.day==7)&(dd.hour>=14)&(dd.hour<16)
            delta=bb.T2_C-dd.T2_C
            conv.append(dict(strategy=p.name,dt_s=dt,N=n,nd=nd,nchi=nchi,cooling_degree_h=float(delta[mask].sum()*dt/3600),lastday_cooling_C=float(delta[end].mean()),transp_mm=dd.transp_mm.sum()))
    pd.DataFrame(conv).to_csv(OUT/'convergence.csv',index=False)
    # Isolate old splitting bias at prescribed soil potential, all other physics same.
    spl=[]
    for dt in [600.,300.,150.,75.]:
        for legacy in [False,True]:
            cc=replace(Config(days=1,fixed_psi=-.3),dt=dt,legacy_split=legacy)
            bb=execute(EV,replace(cc,mode='bare'),'split_bare');dd=execute(EV,cc,'split_full')
            mask=(dd.hour>=14)&(dd.hour<16)
            spl.append(dict(dt_s=dt,scheme='old split' if legacy else 'exact linear',cooling_C=float((bb.T2_C-dd.T2_C)[mask].mean())))
    pd.DataFrame(spl).to_csv(OUT/'splitting_audit.csv',index=False)
    audit=pd.DataFrame(AUDIT);audit.to_csv(OUT/'run_audit.csv',index=False)
    assert np.nanmax(np.abs(audit.water_residual_mm))<1e-8
    assert audit.max_leaf_energy_residual_Wm2.max()<1e-7
    assert audit.max_RH.max()<1, 'Condensation physics required; do not silently clip humidity.'
    metadata={'seconds':time.time()-start,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,'default_config':asdict(Config()),'plants':[asdict(EV),asdict(SC)],'runs':len(AUDIT),'seed':'none: deterministic experiment','sources':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/'shield.py',ROOT/'run_experiments.py',ROOT/'build_report_assets.py',ROOT/'test_model.py']}}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    plots(static,resilience,summary,times,sensitivity,pd.DataFrame(conv),pd.DataFrame(spl),traces)
    from build_report_assets import build
    build()
    print(json.dumps(metadata,indent=2),flush=True)

def plots(static,resilience,s,t,sens,conv,split,traces):
    fig,ax=plt.subplots(1,2,figsize=(10,3.6))
    for d in traces:
        ax[0].plot(d.hour,d.T2_C,label=d.scenario.iloc[0])
    ax[0].set(xlabel='Hour',ylabel='Air temperature at 2 m (°C)');ax[0].legend(fontsize=7)
    for p in [EV,SC]:
        g=static[static.strategy==p.name]
        for psi,ls in [(-.3,'-'),(-1.8,'--')]:
            h=g[g.psi_MPa==psi];ax[1].plot(h.LAI,h.cooling_C,ls+'o',color=COLORS[p.name],label=f'{p.name}, {psi} MPa')
    ax[1].set(xlabel='Leaf area index',ylabel='14–16 h mean cooling (K)');ax[1].legend(fontsize=7)
    fig.tight_layout();savefig('static')
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    for p in [EV,SC]:
        for budget,ls in [(0,'--'),(12,'-')]:
            d=t[(t.strategy==p.name)&(t.LAI==3)&(t.budget_mm==budget)&(t.policy.isin(['none','daily']))]
            daily=d[(d.hour>=14)&(d.hour<16)].groupby('day').cooling_C.mean()
            ax[0].plot(daily.index,daily,ls+'o',color=COLORS[p.name],label=f'{p.name}, {budget} mm')
            ax[1].plot(d.time_h/24,d.psi_MPa,ls,color=COLORS[p.name])
    ax[0].set(xlabel='Heatwave day',ylabel='14–16 h cooling (K)');ax[0].legend(fontsize=7)
    ax[1].set(xlabel='Elapsed days',ylabel='Soil water potential (MPa)')
    fig.tight_layout();savefig('drydown')
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    for p in [EV,SC]:
        for lai,marker in [(1,'o'),(3,'s'),(5,'^')]:
            d=s[(s.strategy==p.name)&(s.LAI==lai)&s.policy.isin(['none','daily'])]
            ax[0].plot(d.irrigation_mm,d.cooling_degree_h,'-'+marker,color=COLORS[p.name],alpha=.5+lai/10,label=f'{p.name}, LAI {lai}')
    ax[0].set(xlabel='Delivered irrigation (mm over 7 days)',ylabel='Daytime cooling service (K h)');ax[0].legend(fontsize=6)
    d=s[(s.LAI==3)&(s.budget_mm==12)]
    for i,p in enumerate([EV,SC]):
        v=d[d.strategy==p.name].set_index('policy').loc[['front','daily','threshold']]
        ax[1].bar(np.arange(3)+(i-.5)*.35,v.lastday_cooling_C,width=.35,color=COLORS[p.name],label=p.name)
    ax[1].set(xticks=range(3),xticklabels=['Front-loaded','Daily','Threshold'],ylabel='Day 7 cooling (K)');ax[1].legend(fontsize=7)
    fig.tight_layout();savefig('budget')
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    rr=pd.DataFrame(resilience)
    for p in [EV,SC]:
        d=rr[rr.strategy==p.name]
        ax[0].plot(d.LAI,d.total_CRI,'-o',color=COLORS[p.name],label=p.name+' total')
        ax[0].plot(d.LAI,d.transp_CRI,'--o',color=COLORS[p.name],label=p.name+' transpiration increment')
    ax[0].set(xlabel='Leaf area index',ylabel='Dry / wet cooling ratio');ax[0].legend(fontsize=6)
    idx=['root_depth_m','tau_s','initial_psi_MPa','heat_increment_C']
    wide=sens.pivot(index=idx,columns='strategy',values='lastday_cooling_C')
    diff=wide[SC.name]-wide[EV.name]
    ax[1].scatter(np.arange(len(diff)),diff,c=np.asarray(wide.index.get_level_values('root_depth_m')),cmap='viridis',s=28)
    ax[1].axhline(0,color='k',lw=.7);ax[1].set(xlabel='Prespecified sensitivity scenario (36)',ylabel='Day 7 cooling: SC minus EV (K)')
    fig.tight_layout();savefig('resilience_sensitivity')
    fig,ax=plt.subplots(figsize=(6,3.5))
    for name,g in split.groupby('scheme'):ax.plot(g.dt_s,g.cooling_C,'o-',label=name)
    ax.set(xlabel='Timestep (s)',ylabel='Fixed-soil cooling (K)');ax.legend();fig.tight_layout();savefig('numerics')

if __name__=='__main__':main()
