"""SHIELD-W: finite-water extension of the user-supplied SHIELD prototype.
Synthetic comparative experiments, not calibrated city predictions. Units SI
except water potential MPa, CO2 ppm and carbon rates umol m-2 leaf s-1.
The physiology is P-Hydro-inspired; no official P-Hydro code is imported.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.linalg import expm

SIGMA=5.670374419e-8
R=8.314462618
CP=1005.
MW=.01801528
P=101325.
XG,WG=np.polynomial.legendre.leggauss(16)

@dataclass(frozen=True)
class Plant:
    name: str='efficient-vulnerable'
    Kp: float=.008
    psi50: float=-1.4
    b: float=1.5
    alpha_cost: float=.08
    gamma_cost: float=1.
    leaf_length: float=.06
    emissivity: float=.96
    absorptivity: float=.72
    ca: float=420.
    br: float=.002
    phi0: float=.087
    gsw_max: float=1.2

EV=Plant()
SC=Plant(name='safe-conservative',Kp=.0055,psi50=-3.)

@dataclass(frozen=True)
class Config:
    days: int=7
    dt: float=600.
    N: int=12
    H: float=12.
    LAI: float=3.
    root_depth: float=.6
    psi_initial: float=-.15
    fixed_psi: float|None=None
    budget: float=12.
    policy: str='daily'  # none, daily, front, threshold
    threshold: float=-.8
    pulse: float=4.
    tau: float=150.
    Kz: float=.35
    wind: float=1.8
    heat: float=0.
    kext: float=.55
    ground_heat: float=300.
    mode: str='full'  # bare, shade, full
    nd: int=81
    nchi: int=61
    spinup_days: int=1
    legacy_split: bool=False
    psi_floor: float=-6.
    max_drawdown: float=6.

# Carsel-Parrish loam; alpha converted from .036 cm-1 to 3.6 m-1.
THR,THS,ALPHA,NVG=.078,.43,3.6,1.56
MVG=1.-1./NVG

def theta_from_psi(psi):
    head=np.abs(np.asarray(psi))/.00980665
    return THR+(THS-THR)*(1+(ALPHA*head)**NVG)**(-MVG)

def psi_from_theta(theta):
    se=np.clip((np.asarray(theta)-THR)/(THS-THR),1e-10,1.)
    return -.00980665/ALPHA*(se**(-1/MVG)-1)**(1/NVG)

def esat(T):
    tc=np.asarray(T)-273.15
    return 611.21*np.exp((18.678-tc/234.5)*tc/(257.14+tc))

def q_from_rh(T,rh):
    e=rh*esat(T)
    return .622*e/(P-.378*e)

def e_from_q(q):
    return q*P/(.622+.378*q)

def latent(T):
    return 2.501e6-2361*(T-273.15)

def weather(hour,c):
    phase=np.sin(2*np.pi*(hour-9)/24)
    T=273.15+31+5.5*phase+c.heat
    rh=np.clip(.58-.18*phase,.28,.90)
    S=900*max(0.,np.sin(np.pi*(hour-6)/12)) if 6<=hour<=18 else 0.
    return T,q_from_rh(T,rh),S,c.wind*(.75+.35*max(phase,0))

def boundary(T,u,p):
    rho=P/(287.05*T)
    nu=1.846e-5*(T/300)**.7/rho
    ka=.02624*(T/300)**.85
    dv=2.5e-5*(T/298.15)**1.81
    re=np.maximum(u,.05)*p.leaf_length/nu
    h=.664*np.sqrt(re)*(nu*rho*CP/ka)**(1/3)*ka/p.leaf_length
    gb=.664*np.sqrt(re)*(nu/dv)**(1/3)*dv/p.leaf_length*P/(R*T)
    return 2*h,gb

def energy(Ta,E,sw,sky,ground,h,p):
    """Vectorized Newton solve; residual returned for numerical auditing."""
    tl=np.broadcast_arrays(Ta,E,sw)[0].copy()
    for _ in range(12):
        rn=p.absorptivity*sw+p.emissivity*SIGMA*(sky**4+ground**4-2*tl**4)
        f=rn-h*(tl-Ta)-E*MW*latent(tl)
        deriv=-8*p.emissivity*SIGMA*tl**3-h+E*MW*2361
        tl-=f/deriv
    rn=p.absorptivity*sw+p.emissivity*SIGMA*(sky**4+ground**4-2*tl**4)
    H=h*(tl-Ta)
    residual=rn-H-E*MW*latent(tl)
    return tl,H,residual

def leaves(T,q,sw,u,psi,p,c,sky,ground,Ecap=np.inf):
    """Global rectangular discrete search, with the zero-transpiration state.
    dpsi candidates use quadratic spacing to resolve low demand. Carbon
    capacity acclimates at every step: this is an equilibrium upper-bound
    closure, not an explicit subdaily acclimation model.
    """
    h,gb=boundary(T,u,p)
    off_tl,off_h,off_r=energy(T,np.zeros_like(T),sw,sky,ground,h,p)
    if c.mode!='full' or sw.max()<1e-8:
        return np.zeros_like(T),off_h,off_tl,float(np.max(np.abs(off_r))),0.
    dmax=c.max_drawdown
    d=dmax*np.linspace(.001,1,c.nd)**2
    # Integral over psi: all dpsi candidates in parallel.
    xp=psi-d[:,None]/2+d[:,None]*XG[None,:]/2
    E=p.Kp*d/2*np.sum(WG[None,:]*.5**(np.abs(xp/p.psi50)**p.b),axis=1)
    tt,hh,res=energy(T[:,None],E[None,:],sw[:,None],sky,ground,h[:,None],p)
    D=(esat(tt)-e_from_q(q[:,None]))/P
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        gtw=E[None,:]/D
        gsw=1/(1/gtw-1/gb[:,None])
        gtc=1/(1/(gsw/1.6)+1/(gb[:,None]/1.37))
        chi=np.linspace(.35,.95,c.nchi)
        ci=chi*p.ca
        A=gtc[:,:,None]*p.ca*(1-chi[None,None,:])
        def arr(k,ea):
            return k*np.exp(ea*(tt-298.15)/(298.15*R*tt))
        gam=arr(42.75,37830)
        km=arr(404.9,79430)*(1+210000/arr(278400,36380))
        numer=ci[None,None,:]*(1-p.br)-(gam+p.br*km)[:,:,None]
        fj=numer/(ci[None,None,:]+2*gam[:,:,None])
        J=4*A/fj
        # Same light response as the supplied prototype; 2.05 umol photons/J.
        jl=4*p.phi0*2.05*sw[:,None,None]
        jm=jl/np.sqrt((jl/J)**2-1)
        profit=A-p.alpha_cost*jm-p.gamma_cost*d[None,:,None]**2
        valid=(D>1e-6)&(gtw>0)&(gtw<.98*gb[:,None])&(gsw>0)&(gsw<=p.gsw_max)&(E[None,:]<=Ecap)
        ok=valid[:,:,None]&(fj>0)&(J>0)&(J<.995*jl)&(2.05*sw[:,None,None]>=10)&np.isfinite(profit)
    F=np.where(ok,profit,-np.inf)
    flat=F.reshape(len(T),-1)
    idx=np.argmax(flat,axis=1)
    jj=idx//c.nchi
    active=flat[np.arange(len(T)),idx]>0
    ev=np.where(active,E[jj],0.)
    hv=np.where(active,hh[np.arange(len(T)),jj],off_h)
    tv=np.where(active,tt[np.arange(len(T)),jj],off_tl)
    residual=float(max(np.max(np.abs(res)),np.max(np.abs(off_r))))
    return ev,hv,tv,residual,float(np.sum(active & (jj==c.nd-1)))

def transport(c,z):
    dz=c.H/c.N
    k=c.Kz+(z/c.H)**1.3
    face=(k[1:]+k[:-1])/2/dz**2
    L=np.diag(-np.r_[face,0]-np.r_[0,face])+np.diag(face,1)+np.diag(face,-1)
    rates=1/(c.tau*(1.25-.5*z/c.H))
    rates[-1]+=1/180.
    A=L-np.diag(rates)
    M=expm(A*c.dt)
    B=np.linalg.solve(A,M-np.eye(c.N))
    return M,B,L,rates

def run(p=EV,c=Config()):
    """Returns step-level data and diagnostics; every record spans exactly dt."""
    assert 86400 % c.dt==0 and c.policy in ('none','daily','front','threshold')
    assert c.mode in ('bare','shade','full')
    dz=c.H/c.N
    z=(np.arange(c.N)+.5)*dz
    lai=0. if c.mode=='bare' else c.LAI
    profile=(z/c.H)**1.4*(1-z/c.H)
    lad=lai*profile/(profile.sum()*dz)
    area=lad*dz
    above=np.cumsum(area[::-1])[::-1]-area
    trans=np.exp(-c.kext*above)
    # Exactly conservative interception across finite cells.
    swleaf=np.divide(trans*(1-np.exp(-c.kext*area)),area,out=np.zeros_like(area),where=area>0)
    M,B,L,rates=transport(c,z)
    T0,q0,_,_=weather(0,c)
    T=np.full(c.N,T0); q=np.full(c.N,q0)
    theta0=float(theta_from_psi(c.psi_initial))
    theta=theta0; floor=float(theta_from_psi(c.psi_floor))
    budget_left=c.budget if c.policy!='none' else 0.
    water=irr=drain=0.
    maxres=0.; boundary_count=0.; supersat=0.; cap_hits=0
    records=[]
    steps=int(86400/c.dt)
    for it in range(-c.spinup_days*steps,c.days*steps):
        spin=it<0
        hour=((it+.5)*c.dt/3600)%24
        day=max(0,it//steps)
        # Irrigation once daily at 06:00, before the interval's uptake.
        apply=0.
        if not spin and it%steps==int(6*3600/c.dt) and c.fixed_psi is None:
            if c.policy=='front' and day==0: apply=budget_left
            elif c.policy=='daily': apply=min(budget_left,c.budget/c.days)
            elif c.policy=='threshold' and psi_from_theta(theta)<c.threshold: apply=min(budget_left,c.pulse)
            budget_left-=apply; irr+=apply
            theta+=apply/(1000*c.root_depth)
            overflow=max(0.,theta-THS)*1000*c.root_depth
            theta=min(theta,THS); drain+=overflow
        psi=c.fixed_psi if c.fixed_psi is not None else (c.psi_initial if spin else float(psi_from_theta(theta)))
        Tbg,qbg,SW,wind=weather(hour,c)
        sky=Tbg-12; ground=Tbg+8*(SW/900)*np.exp(-c.kext*lai)
        available=max(0.,(theta-floor)*1000*c.root_depth)
        cap=available/(MW*max(lai,1e-12)*c.dt) if not spin and c.fixed_psi is None else np.inf
        E,H,Tl,res,bc=leaves(T,q,SW*swleaf,wind*np.exp(-.28*(above+.5*area)),psi,p,c,sky,ground,cap)
        maxres=max(maxres,res); boundary_count+=bc
        demand=float(np.sum(E*area)*MW*c.dt)
        if not spin and c.fixed_psi is None:
            assert demand<=available+1e-10
            theta-=demand/(1000*c.root_depth)
            cap_hits+=int(available<1e-6)
        rho=P/(287.05*T*(1+.61*q))
        st=lad*H/(rho*CP)
        st[0]+=c.ground_heat*(SW/900)*np.exp(-c.kext*lai)/(rho[0]*CP*dz)
        sq=lad*E*MW/rho
        if c.legacy_split:
            # Audit old source-then-relax update, same revised physics otherwise.
            diff=np.linalg.inv(np.eye(c.N)-c.dt*L)
            T=diff@(Tbg+(T+c.dt*st-Tbg)*np.exp(-c.dt*rates))
            q=diff@(qbg+(q+c.dt*sq-qbg)*np.exp(-c.dt*rates))
        else:
            T=Tbg+M@(T-Tbg)+B@st
            q=qbg+M@(q-qbg)+B@sq
        rh=e_from_q(q)/esat(T)
        supersat=max(supersat,float(np.max(rh)))
        if not spin:
            water+=demand
            records.append(dict(day=day+1,hour=hour,time_h=(it+.5)*c.dt/3600,T2_C=float(np.interp(2,z,T)-273.15),q2=float(np.interp(2,z,q)),psi_MPa=psi,theta_end=theta,transp_mm=demand,irrigation_mm=apply,Tleaf_C=float(np.sum(Tl*area)/max(lai,1e-12)-273.15) if lai else np.nan,latent_Wm2=float(np.sum(E*MW*latent(Tl)*area))))
    residual=1000*c.root_depth*(theta-theta0)+water+drain-irr if c.fixed_psi is None else np.nan
    return pd.DataFrame(records),dict(water_residual_mm=residual,max_leaf_energy_residual_Wm2=maxres,max_RH=supersat,hydraulic_bound_hits=boundary_count,storage_floor_hits=cap_hits,transp_mm=water,irrigation_mm=irr,drainage_mm=drain,theta_final=theta)
