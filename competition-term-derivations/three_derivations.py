# """
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

# ============================================================
# OPERADORES PARA 3 QUBITS (dimensão 8)
# ============================================================
I = np.eye(2, dtype=complex)
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
sz = np.array([[1,0],[0,-1]], dtype=complex)
sm = np.array([[0,0],[1,0]], dtype=complex)
sp = sm.conj().T

def kron3(A, B, C):
    return np.kron(A, np.kron(B, C))

I3 = np.eye(8, dtype=complex)
sm1 = kron3(sm, I, I)
sm2 = kron3(I, sm, I)
sm3 = kron3(I, I, sm)
sp1 = sm1.conj().T
sp2 = sm2.conj().T
sp3 = sm3.conj().T
sx1 = kron3(sx, I, I)
sx2 = kron3(I, sx, I)
sx3 = kron3(I, I, sx)

H0 = 0.5 * (sx1 + sx2 + sx3)
GAMMA = 0.2
Ls = [np.sqrt(GAMMA)*sm1, np.sqrt(GAMMA)*sm2, np.sqrt(GAMMA)*sm3]

psi0 = np.array([1,1])/np.sqrt(2)
psi0_3 = np.kron(psi0, np.kron(psi0, psi0))
RHO0 = np.outer(psi0_3, psi0_3.conj())

T_MAX = 20.0
DT = 0.01
STEPS = int(T_MAX/DT)
tlist = np.linspace(0, T_MAX, STEPS)

def dag(A): return A.conj().T
def proj(rho):
    rho = (rho + dag(rho))/2
    v, U = np.linalg.eigh(rho)
    v = np.clip(v.real, 0, None)
    s = v.sum()
    return U @ np.diag((v/s).astype(complex)) @ dag(U)

def lindblad_drho(rho, H, L_list):
    d = -1j*(H@rho - rho@H)
    for L in L_list:
        d += L@rho@dag(L) - 0.5*(dag(L)@L@rho + rho@dag(L)@L)
    return d

def rk4(rho, dt, f):
    k1 = f(rho)
    k2 = f(rho + dt/2*k1)
    k3 = f(rho + dt/2*k2)
    k4 = f(rho + dt*k3)
    return proj(rho + dt/6*(k1+2*k2+2*k3+k4))

def purity(rho): return np.real(np.trace(rho@rho))

# ============================================================
# CAMINHO A: CAMPO MÉDIO
# ============================================================
def mean_field_drho(rho, lam_mf):
    drho = lindblad_drho(rho, H0, Ls)
    rho2 = rho @ rho
    comp = rho2 - rho * np.trace(rho2)
    drho += lam_mf * comp
    return drho

# ============================================================
# CAMINHO B: COLISÃO ADAPTATIVA
# ============================================================
def build_V():
    V = np.zeros((64,64), dtype=complex)
    for op in [sx1, sx2, sx3]:
        V += np.kron(op, op)
    return V

V_total = build_V()

def collision_drho_effective(rho, alpha, g):
    drho = lindblad_drho(rho, H0, Ls)
    d = 8
    rho_anc = alpha * rho + (1-alpha) * np.eye(d)/d
    rho_anc = proj(rho_anc)
    rho_tot = np.kron(rho, rho_anc)
    V2rho = V_total @ rho_tot @ dag(V_total)
    drho_nl = np.trace(V2rho.reshape(d,d,d,d), axis1=1, axis2=3) - rho
    norm = np.linalg.norm(drho_nl)
    if norm > 1e-12:
        drho_nl /= norm
    drho += g**2 * alpha * drho_nl
    return drho

# ============================================================
# CAMINHO C: ELIMINAÇÃO ADIABÁTICA
# ============================================================
kappa = 2.0
MEMORY = 10

def adiabatic_drho(rho, g, memory_rhos, dt):
    drho = lindblad_drho(rho, H0, Ls)
    if len(memory_rhos) == 0:
        return drho
    tau_max = min(len(memory_rhos), MEMORY)
    L2 = np.zeros_like(rho)
    for k in range(tau_max):
        tau = (k+1)*dt
        C_tau = g**2 * np.exp(-kappa * tau)
        rho_past = memory_rhos[-(k+1)]
        for sp_i, sm_i in [(sp1,sm1), (sp2,sm2), (sp3,sm3)]:
            L2 += C_tau * (sp_i @ rho @ sm_i @ rho_past - 0.5*(sp_i@sm_i@rho@rho_past + rho_past@rho@sp_i@sm_i))
            L2 += C_tau * (sm_i @ rho @ sp_i @ rho_past - 0.5*(sm_i@sp_i@rho@rho_past + rho_past@rho@sm_i@sp_i))
    drho += np.real(L2)
    return drho

# ============================================================
# SIMULAÇÕES
# ============================================================
print("▶ Simulando Lindblad puro...")
rho = RHO0.copy()
pur_ref = []
for _ in range(STEPS):
    pur_ref.append(purity(rho))
    rho = rk4(rho, DT, lambda r: lindblad_drho(r, H0, Ls))
pur_ref = np.array(pur_ref)

print("▶ Simulando Campo Médio (λ=0.6)...")
rho = RHO0.copy()
pur_A = []
for _ in range(STEPS):
    pur_A.append(purity(rho))
    rho = rk4(rho, DT, lambda r: mean_field_drho(r, 0.6))
pur_A = np.array(pur_A)

print("▶ Simulando Colisão Adaptativa (α=0.7, g=0.8)...")
rho = RHO0.copy()
pur_B = []
for _ in range(STEPS):
    pur_B.append(purity(rho))
    rho = rk4(rho, DT, lambda r: collision_drho_effective(r, 0.7, 0.8))
pur_B = np.array(pur_B)

print("▶ Simulando Adiabático (g=0.6)...")
rho = RHO0.copy()
pur_C = []
memory = []
for _ in range(STEPS):
    pur_C.append(purity(rho))
    rho_new = rk4(rho, DT, lambda r: adiabatic_drho(r, 0.6, memory, DT))
    memory.append(rho.copy())
    if len(memory) > MEMORY*2:
        memory.pop(0)
    rho = rho_new
pur_C = np.array(pur_C)

# ============================================================
# GRÁFICO
# ============================================================
plt.figure(figsize=(10,6))
plt.plot(tlist, pur_ref, 'k--', label='Lindblad puro', lw=2)
plt.plot(tlist, pur_A, 'b-', label='A: Campo médio (λ=0.6)', lw=2)
plt.plot(tlist, pur_B, 'g-', label='B: Colisão adaptativa (α=0.7, g=0.8)', lw=2)
plt.plot(tlist, pur_C, 'r-', label='C: Adiabático (g=0.6)', lw=2)
plt.xlabel('t (u.a.)', fontsize=12)
plt.ylabel('Pureza P(ρ)', fontsize=12)
plt.title('3 qubits – Comparação das três derivações', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparacao_3qubits.png', dpi=120)
plt.show()

print("\n✅ Simulação concluída! Gráfico salvo como 'comparacao_3qubits.png'")
