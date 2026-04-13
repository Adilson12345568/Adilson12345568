"""
TRÊS DERIVAÇÕES SIMULTÂNEAS DA ORIGEM FÍSICA DO TERMO DE COMPETIÇÃO
  A: Campo médio de N corpos — aproximação fatorada
  B: Colisão adaptativa — ambiente que aprende
  C: Hamiltoniano efetivo — eliminação adiabática de 2ª ordem
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import expm
import warnings
warnings.filterwarnings('ignore')

rng = np.random.default_rng(42)

# ══════════════════════════════════════════════════════════════
# OPERADORES BASE (qubit)
# ══════════════════════════════════════════════════════════════
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)
sm = np.array([[0,0],[1,0]], dtype=complex)   # σ−
sp = sm.conj().T                               # σ+
I2 = np.eye(2, dtype=complex)

def dag(A): return A.conj().T
def proj(rho):
    rho=(rho+dag(rho))/2; v,U=np.linalg.eigh(rho)
    v=np.clip(v.real,0,None); s=v.sum()
    return U@np.diag((v/s).astype(complex))@dag(U)

def lindblad_drho(rho, H, Ls):
    d = -1j*(H@rho - rho@H)
    for L in Ls:
        d += L@rho@dag(L) - 0.5*(dag(L)@L@rho + rho@dag(L)@L)
    return d

def rk4(rho, dt, f):
    k1=f(rho); k2=f(rho+dt/2*k1); k3=f(rho+dt/2*k2); k4=f(rho+dt*k3)
    return proj(rho + dt/6*(k1+2*k2+2*k3+k4))

def purity(rho): return np.real(np.trace(rho@rho))

# ══════════════════════════════════════════════════════════════
# PARÂMETROS GLOBAIS
# ══════════════════════════════════════════════════════════════
H0    = 0.5*sx          # Hamiltoniano do sistema
GAMMA = 0.2             # decoerência
Ls    = [np.sqrt(GAMMA)*sm]   # Lindblad padrão
T_MAX = 20.0
DT    = 0.01
STEPS = int(T_MAX/DT)
tlist = np.linspace(0, T_MAX, STEPS)

psi0  = np.array([1,1],dtype=complex)/np.sqrt(2)
RHO0  = np.outer(psi0, psi0.conj())

# ══════════════════════════════════════════════════════════════
# REFERÊNCIA: Lindblad puro (sem competição)
# ══════════════════════════════════════════════════════════════
rho = RHO0.copy()
pur_ref=[]
for _ in range(STEPS):
    pur_ref.append(purity(rho))
    rho = rk4(rho, DT, lambda r: lindblad_drho(r, H0, Ls))
pur_ref=np.array(pur_ref)

print("▶ CAMINHO A — Campo Médio de N corpos …")
# ══════════════════════════════════════════════════════════════
# CAMINHO A: CAMPO MÉDIO
def mean_field_drho(rho, lam_mf):
    drho = lindblad_drho(rho, H0, Ls)
    sm_mean = np.trace(sm @ rho)
    sp_mean = np.trace(sp @ rho)
    H_mf = lam_mf * (sp_mean * sm + sm_mean * sp)
    drho += -1j*(H_mf@rho - rho@H_mf)
    rho2 = rho @ rho
    comp_mf = rho2 - rho * np.trace(rho2)
    drho += lam_mf**2 * comp_mf
    return drho

lams_mf = [0.0, 0.3, 0.6, 1.0, 1.5]
traj_A = {}
for lam in lams_mf:
    rho=RHO0.copy(); pp=[]
    for _ in range(STEPS):
        pp.append(purity(rho))
        rho=rk4(rho, DT, lambda r,l=lam: mean_field_drho(r, l))
    traj_A[lam] = np.array(pp)

print("   ✓ Campo médio concluído")

print("▶ CAMINHO B — Colisão Adaptativa …")
g_coll  = 0.8
alpha   = 0.7
V_int   = np.kron(sx, sx)

def collision_drho_effective(rho, alpha=0.7, g=0.8):
    drho = lindblad_drho(rho, H0, Ls)
    rho_anc = alpha*rho + (1-alpha)*np.array([[1,0],[0,0]],dtype=complex)
    rho_anc = proj(rho_anc)
    rho_tot = np.kron(rho, rho_anc)
    V2rho = V_int @ rho_tot @ dag(V_int)
    drho_nl = np.trace(V2rho.reshape(2,2,2,2), axis1=1, axis2=3) - rho
    norm = np.linalg.norm(drho_nl)
    if norm > 1e-12:
        drho_nl /= norm
    drho += g**2 * alpha * drho_nl
    return drho

alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
traj_B = {}
for al in alphas:
    rho=RHO0.copy(); pp=[]
    for _ in range(STEPS):
        pp.append(purity(rho))
        rho=rk4(rho, DT, lambda r,a=al: collision_drho_effective(r, alpha=a))
    traj_B[al] = np.array(pp)

print("▶ CAMINHO C — Hamiltoniano Efetivo (eliminação adiabática) …")
kappa   = 2.0
MEMORY  = 10

def adiabatic_drho(rho, g, kappa, memory_rhos):
    drho = lindblad_drho(rho, H0, Ls)
    if len(memory_rhos) > 0:
        tau_max = min(len(memory_rhos), MEMORY)
        L2 = np.zeros_like(rho)
        for k in range(tau_max):
            tau = (k+1)*DT
            C_tau = g**2 * np.exp(-kappa * tau)
            rho_past = memory_rhos[-(k+1)]
            L2 += C_tau * (sp@rho@sm@rho_past - 0.5*(sp@sm@rho@rho_past + rho_past@rho@sp@sm))
            L2 += C_tau * (sm@rho@sp@rho_past - 0.5*(sm@sp@rho@rho_past + rho_past@rho@sm@sp))
        drho += np.real(L2)
    return drho

g_vals = [0.0, 0.2, 0.4, 0.6, 0.8]
traj_C = {}
for g in g_vals:
    rho=RHO0.copy(); pp=[]; memory=[]
    for _ in range(STEPS):
        pp.append(purity(rho))
        rho = rk4(rho, DT, lambda r: adiabatic_drho(r, g, kappa, memory))
        memory.append(rho.copy())
        if len(memory) > MEMORY*2: memory.pop(0)
    traj_C[g] = np.array(pp)

print("▶ COMPARAÇÃO DAS TRÊS DERIVAÇÕES …")
def original_competition(rho, lam=1.0):
    diag=np.real(np.diag(rho))
    pop=np.diag(diag*(diag-np.mean(diag)))
    coh=rho@rho-rho
    comp=pop+coh
    norm=np.linalg.norm(comp)
    return lam*comp/(1e-12+norm)

def original_drho(rho, lam=1.0):
    return lindblad_drho(rho, H0, Ls) + original_competition(rho, lam)

rho=RHO0.copy(); pp_orig=[]
for _ in range(STEPS):
    pp_orig.append(purity(rho))
    rho=rk4(rho, DT, lambda r: original_drho(r, 1.0))
pp_orig=np.array(pp_orig)

rho=RHO0.copy(); pp_A=[]
for _ in range(STEPS):
    pp_A.append(purity(rho))
    rho=rk4(rho, DT, lambda r: mean_field_drho(r, 0.6))
pp_A=np.array(pp_A)

rho=RHO0.copy(); pp_B=[]
for _ in range(STEPS):
    pp_B.append(purity(rho))
    rho=rk4(rho, DT, lambda r: collision_drho_effective(r, alpha=0.7))
pp_B=np.array(pp_B)

rho=RHO0.copy(); pp_C=[]; memory=[]
for _ in range(STEPS):
    pp_C.append(purity(rho))
    rho=rk4(rho, DT, lambda r: adiabatic_drho(r, 0.6, kappa, memory))
    memory.append(rho.copy())
    if len(memory)>MEMORY*2: memory.pop(0)
pp_C=np.array(pp_C)

print(f"Pureza final: Original={pp_orig[-1]:.4f}  A={pp_A[-1]:.4f}  B={pp_B[-1]:.4f}  C={pp_C[-1]:.4f}")

# ══════════════════════════════════════════════════════════════
# FIGURA
# ══════════════════════════════════════════════════════════════
plt.figure(figsize=(10,6))
plt.plot(tlist, pp_orig, 'k-', label='Original (competição pura)', lw=2)
plt.plot(tlist, pp_A, 'b--', label='A: Campo médio (λ=0.6)', lw=2)
plt.plot(tlist, pp_B, 'g-.', label='B: Colisão adaptativa (α=0.7)', lw=2)
plt.plot(tlist, pp_C, 'r:', label='C: Adiabático (g=0.6)', lw=2)
plt.plot(tlist, pur_ref, 'k:', alpha=0.5, label='Lindblad puro', lw=1.5)
plt.xlabel('t (u.a.)')
plt.ylabel('Pureza P(ρ)')
plt.title('Comparação das três derivações – 1 qubit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparacao_derivacoes.png', dpi=120)
plt.show()

print("\n✅ Figura salva: comparacao_derivacoes.png")
