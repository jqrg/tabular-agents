# -*- coding: utf-8 -*-
"""
Generador del dataset de evaluación del TFM.
10 problemas de regresión: 3 de control (con fórmula) + 7 sin fórmula reconocible.

Anexo de documentación: deja constancia de cómo se generaron los prompts y el
ground truth. Los ficheros vigentes ya están en prompts/ (son los artefactos
congelados de la evaluación) y no hace falta volver a ejecutarlo; OUT_DIR
apunta al entorno donde se generó originalmente.

Reglas aplicadas a TODOS los problemas:
  R1. El punto de consulta (query) NUNCA se incluye en el CSV (sin data leakage).
  R2. Números redondeados a 6 cifras significativas.
  R3. Query dentro del rango de los datos (interpolación): se muestrea en el
      20-80% del rango de cada variable de entrada.

Salida:
  prompts_tfm/prompt_01_... .txt  (formato idéntico al de los prompts originales)
  prompts_tfm/soluciones.txt      (ground truth de cada query, para el harness)
"""
import numpy as np
import os

OUT_DIR = "/mnt/user-data/outputs/prompts_tfm"
os.makedirs(OUT_DIR, exist_ok=True)

N_ROWS = 100          # filas del CSV (el query va aparte, regla R1)
SIG = 6               # cifras significativas (regla R2)

def f6(x):
    """Formatea a 6 cifras significativas."""
    return "%.*g" % (SIG, x)

def sample_interior(rng, lo, hi):
    """Muestrea en el 20-80% del rango (regla R3: interpolación)."""
    span = hi - lo
    return rng.uniform(lo + 0.2 * span, hi + -0.2 * span + 0.0) if False else rng.uniform(lo + 0.2*span, hi - 0.2*span)

def build_prompt(outputs, inputs_names, query_vals, header, rows):
    """Construye el prompt con el formato exacto de los originales."""
    if len(outputs) == 1:
        pregunta = "la " + outputs[0]
    else:
        pregunta = " y ".join("la " + o for o in outputs)
    lines = ["Quiero predecir " + pregunta + " para"]
    for name, val in zip(inputs_names, query_vals):
        lines.append("    " + name + " = " + f6(val))
    lines.append(" ")
    lines.append("    Basándome en estos datos previos:")
    lines.append(" ")
    lines.append(",".join(header))
    for r in rows:
        lines.append(",".join(f6(v) for v in r))
    return "\n".join(lines)

soluciones = []

def save_problem(idx, slug, outputs, inputs_names, query_vals, header, rows, truth, nota):
    fname = "prompt_%02d_%s.txt" % (idx, slug)
    with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
        f.write(build_prompt(outputs, inputs_names, query_vals, header, rows))
    soluciones.append({
        "archivo": fname,
        "inputs": {n: f6(v) for n, v in zip(inputs_names, query_vals)},
        "outputs_correctos": {o: f6(t) for o, t in zip(outputs, truth)},
        "nota": nota,
    })
    print("OK  %-38s -> %s" % (fname, " | ".join("%s=%s" % (o, f6(t)) for o, t in zip(outputs, truth))))

# =====================================================================
# 1. TRIANGULACIÓN (control: fórmula con parámetros latentes)
#    Anclas OCULTAS (no aparecen en el prompt): A, B, C.
# =====================================================================
rng = np.random.default_rng(101)
ANCLAS = np.array([[120.0, 80.0], [950.0, 150.0], [420.0, 880.0]])

def dists(p):
    return np.sqrt(((ANCLAS - p) ** 2).sum(axis=1))

rows = []
for _ in range(N_ROWS):
    p = rng.uniform(0, 1000, size=2)
    d = dists(p)
    rows.append([d[0], d[1], d[2], p[0], p[1]])
q_pos = np.array([sample_interior(rng, 0, 1000), sample_interior(rng, 0, 1000)])
q_d = dists(q_pos)
save_problem(
    1, "triangulacion",
    ["Posicion X", "Posicion Y"],
    ["Distance A", "Distance B", "Distance C"],
    [q_d[0], q_d[1], q_d[2]],
    ["Distance A", "Distance B", "Distance C", "Posicion X", "Posicion Y"],
    rows, [q_pos[0], q_pos[1]],
    "Control B: formula existe pero requiere anclas ocultas A=(120,80) B=(950,150) C=(420,880). LLM directo no puede aplicarla.",
)

# =====================================================================
# 2. GAS IDEAL (control: memorización pura, plug-and-play)
#    T = P*V/(n*R), R = 8.314. Se muestrea T y se deriva V.
# =====================================================================
rng = np.random.default_rng(102)
R_GAS = 8.314
rows = []
for _ in range(N_ROWS):
    P = rng.uniform(100, 480)
    n = rng.uniform(0.05, 1.0)
    T = rng.uniform(250, 1200)
    V = n * R_GAS * T / P
    rows.append([P, V, n, T])
qP = sample_interior(rng, 100, 480)
qn = sample_interior(rng, 0.05, 1.0)
qT = sample_interior(rng, 250, 1200)
qV = qn * R_GAS * qT / qP
save_problem(
    2, "gas_ideal",
    ["Temperatura"],
    ["Presion", "Volumen", "Moles"],
    [qP, qV, qn],
    ["Presion", "Volumen", "Moles", "Temperatura"],
    rows, [qT],
    "Control A: T = P*V/(n*R) con R=8.314. Formula plug-and-play: mide el techo por memorizacion.",
)

# =====================================================================
# 3. CINEMÁTICA INVERSA 2 ESLABONES (control: fórmula con ambigüedad de rama)
#    Se genera con cinemática directa y rama FIJA (codo positivo).
# =====================================================================
rng = np.random.default_rng(103)
rows = []
for _ in range(N_ROWS):
    L1 = rng.uniform(0.30, 0.80)
    L2 = rng.uniform(0.20, 0.50)
    t1 = rng.uniform(-90, 90)      # grados
    t2 = rng.uniform(15, 160)      # rama codo positivo, lejos de singularidad
    r1, r2 = np.radians(t1), np.radians(t2)
    x = L1 * np.cos(r1) + L2 * np.cos(r1 + r2)
    y = L1 * np.sin(r1) + L2 * np.sin(r1 + r2)
    rows.append([L1, L2, x, y, t1, t2])
qL1 = sample_interior(rng, 0.30, 0.80)
qL2 = sample_interior(rng, 0.20, 0.50)
qt1 = sample_interior(rng, -90, 90)
qt2 = sample_interior(rng, 15, 160)
qr1, qr2 = np.radians(qt1), np.radians(qt2)
qx = qL1 * np.cos(qr1) + qL2 * np.cos(qr1 + qr2)
qy = qL1 * np.sin(qr1) + qL2 * np.sin(qr1 + qr2)
save_problem(
    3, "cinematica_inversa",
    ["Angulo articulacion 1", "Angulo articulacion 2"],
    ["Longitud eslabon 1", "Longitud eslabon 2", "Posicion X objetivo", "Posicion Y objetivo"],
    [qL1, qL2, qx, qy],
    ["Longitud eslabon 1", "Longitud eslabon 2", "Posicion X objetivo", "Posicion Y objetivo",
     "Angulo articulacion 1", "Angulo articulacion 2"],
    rows, [qt1, qt2],
    "Control C: formula IK clasica pero con 2 soluciones; los datos siguen SOLO la rama codo positivo (t2 en [15,160]).",
)

# =====================================================================
# 4. COMPOSICIÓN NO RECONOCIBLE (sin fórmula, matemática)
#    f = sin(2A)*exp(-B/5) + log(1+C^2)*sqrt(D) + 0.3*A*C
# =====================================================================
rng = np.random.default_rng(104)
def f4(A, B, C, D):
    return np.sin(2*A)*np.exp(-B/5.0) + np.log(1+C**2)*np.sqrt(D) + 0.3*A*C
rows = []
for _ in range(N_ROWS):
    A = rng.uniform(0.5, 5); B = rng.uniform(1, 10)
    C = rng.uniform(0.1, 3); D = rng.uniform(1, 8)
    rows.append([A, B, C, D, f4(A, B, C, D)])
qA = sample_interior(rng, 0.5, 5); qB = sample_interior(rng, 1, 10)
qC = sample_interior(rng, 0.1, 3); qD = sample_interior(rng, 1, 8)
save_problem(
    4, "composicion",
    ["Resultado"],
    ["Variable A", "Variable B", "Variable C", "Variable D"],
    [qA, qB, qC, qD],
    ["Variable A", "Variable B", "Variable C", "Variable D", "Resultado"],
    rows, [f4(qA, qB, qC, qD)],
    "Sin formula: f = sin(2A)*exp(-B/5) + log(1+C^2)*sqrt(D) + 0.3*A*C. Nombres genericos.",
)

# =====================================================================
# 5. INTERACCIÓN MULTIPLICATIVA, 2 SALIDAS (sin fórmula, matemática)
#    P = A*C - B^2 + 2*tanh(A+B)
#    Q = C*exp(-|A-B|/2) + A*B/3
# =====================================================================
rng = np.random.default_rng(105)
def f5(A, B, C):
    P = A*C - B**2 + 2*np.tanh(A+B)
    Q = C*np.exp(-np.abs(A-B)/2.0) + A*B/3.0
    return P, Q
rows = []
for _ in range(N_ROWS):
    A = rng.uniform(-3, 3); B = rng.uniform(-3, 3); C = rng.uniform(0.5, 4)
    P, Q = f5(A, B, C)
    rows.append([A, B, C, P, Q])
qA = sample_interior(rng, -3, 3); qB = sample_interior(rng, -3, 3); qC = sample_interior(rng, 0.5, 4)
qP, qQ = f5(qA, qB, qC)
save_problem(
    5, "interaccion_dos_salidas",
    ["Salida P", "Salida Q"],
    ["Variable A", "Variable B", "Variable C"],
    [qA, qB, qC],
    ["Variable A", "Variable B", "Variable C", "Salida P", "Salida Q"],
    rows, [qP, qQ],
    "Sin formula, 2 salidas: P = A*C - B^2 + 2*tanh(A+B); Q = C*exp(-|A-B|/2) + A*B/3.",
)

# =====================================================================
# 6. RÉGIMEN CONDICIONAL (sin fórmula, discontinuidad oculta)
#    Si A+B > 0:  y = 0.5*A^2 + C
#    Si A+B <= 0: y = -1.5*B + C^2
# =====================================================================
rng = np.random.default_rng(106)
def f6c(A, B, C):
    return np.where(A + B > 0, 0.5*A**2 + C, -1.5*B + C**2)
rows = []
for _ in range(N_ROWS):
    A = rng.uniform(-5, 5); B = rng.uniform(-5, 5); C = rng.uniform(0.5, 5)
    rows.append([A, B, C, float(f6c(A, B, C))])
# query en la rama A+B>0 (documentado en soluciones)
while True:
    qA = sample_interior(rng, -5, 5); qB = sample_interior(rng, -5, 5)
    if qA + qB > 0.5:
        break
qC = sample_interior(rng, 0.5, 5)
save_problem(
    6, "regimen_condicional",
    ["Resultado"],
    ["Variable A", "Variable B", "Variable C"],
    [qA, qB, qC],
    ["Variable A", "Variable B", "Variable C", "Resultado"],
    rows, [float(f6c(qA, qB, qC))],
    "Sin formula, piecewise oculto: si A+B>0 -> 0.5*A^2+C; si no -> -1.5*B+C^2. Query en rama A+B>0.",
)

# =====================================================================
# 7. TIEMPO DE COCCIÓN (sin fórmula, empírico plausible)
#    t = 15*peso^0.75 * (185/T)^1.8 * (1 + 0.004*humedad*grosor) + 6*sqrt(grosor)
# =====================================================================
rng = np.random.default_rng(107)
def f7(peso, T, hum, gro):
    return 15*peso**0.75 * (185.0/T)**1.8 * (1 + 0.004*hum*gro) + 6*np.sqrt(gro)
rows = []
for _ in range(N_ROWS):
    peso = rng.uniform(0.5, 4); T = rng.uniform(140, 220)
    hum = rng.uniform(20, 80); gro = rng.uniform(2, 12)
    rows.append([peso, T, hum, gro, f7(peso, T, hum, gro)])
qp = sample_interior(rng, 0.5, 4); qT = sample_interior(rng, 140, 220)
qh = sample_interior(rng, 20, 80); qg = sample_interior(rng, 2, 12)
save_problem(
    7, "tiempo_coccion",
    ["Tiempo de coccion"],
    ["Peso de la pieza", "Temperatura del horno", "Humedad inicial", "Grosor"],
    [qp, qT, qh, qg],
    ["Peso de la pieza", "Temperatura del horno", "Humedad inicial", "Grosor", "Tiempo de coccion"],
    rows, [f7(qp, qT, qh, qg)],
    "Empirico inventado (no corresponde a ninguna formula real): t = 15*peso^0.75*(185/T)^1.8*(1+0.004*hum*grosor)+6*sqrt(grosor).",
)

# =====================================================================
# 8. CONSUMO ELÉCTRICO (sin fórmula, empírico plausible)
#    kWh = (pot/1000)*ciclos*4.33*(0.8+0.015*antig)*(1+0.0008*(temp-22)^2)*1.1
# =====================================================================
rng = np.random.default_rng(108)
def f8(pot, temp, antig, ciclos):
    return (pot/1000.0)*ciclos*4.33*(0.8+0.015*antig)*(1+0.0008*(temp-22)**2)*1.1
rows = []
for _ in range(N_ROWS):
    pot = rng.uniform(500, 3000); temp = rng.uniform(5, 35)
    antig = rng.uniform(0, 15); cic = rng.uniform(1, 21)
    rows.append([pot, temp, antig, cic, f8(pot, temp, antig, cic)])
qpo = sample_interior(rng, 500, 3000); qte = sample_interior(rng, 5, 35)
qan = sample_interior(rng, 0, 15); qci = sample_interior(rng, 1, 21)
save_problem(
    8, "consumo_electrico",
    ["Consumo mensual"],
    ["Potencia nominal", "Temperatura ambiente", "Antiguedad", "Ciclos por semana"],
    [qpo, qte, qan, qci],
    ["Potencia nominal", "Temperatura ambiente", "Antiguedad", "Ciclos por semana", "Consumo mensual"],
    rows, [f8(qpo, qte, qan, qci)],
    "Empirico inventado: kWh = (pot/1000)*ciclos*4.33*(0.8+0.015*antig)*(1+0.0008*(temp-22)^2)*1.1.",
)

# =====================================================================
# 9. LORENZ DISCRETIZADO (sin fórmula cerrada; mapa determinista un paso)
#    sigma=10, rho=28, beta=8/3. RK4. Estado(t) -> Estado(t + 0.1)
# =====================================================================
rng = np.random.default_rng(109)
SIGMA, RHO, BETA = 10.0, 28.0, 8.0/3.0
def lorenz_deriv(s):
    x, y, z = s
    return np.array([SIGMA*(y-x), x*(RHO-z)-y, x*y-BETA*z])
def rk4_step(s, dt):
    k1 = lorenz_deriv(s)
    k2 = lorenz_deriv(s + dt/2*k1)
    k3 = lorenz_deriv(s + dt/2*k2)
    k4 = lorenz_deriv(s + dt*k3)
    return s + dt/6*(k1 + 2*k2 + 2*k3 + k4)
def advance(s, T=0.1, dt=0.01):
    for _ in range(int(round(T/dt))):
        s = rk4_step(s, dt)
    return s
# burn-in hacia el atractor y muestreo descorrelacionado
s = np.array([1.0, 1.0, 20.0])
for _ in range(3000):
    s = rk4_step(s, 0.01)
states = []
for _ in range(N_ROWS + 1):
    for _ in range(53):          # descorrelacionar muestras
        s = rk4_step(s, 0.01)
    states.append(s.copy())
rows = []
for st in states[:N_ROWS]:
    nxt = advance(st.copy())
    rows.append([st[0], st[1], st[2], nxt[0], nxt[1], nxt[2]])
q_state = states[N_ROWS]
q_next = advance(q_state.copy())
save_problem(
    9, "lorenz",
    ["X siguiente", "Y siguiente", "Z siguiente"],
    ["X actual", "Y actual", "Z actual"],
    [q_state[0], q_state[1], q_state[2]],
    ["X actual", "Y actual", "Z actual", "X siguiente", "Y siguiente", "Z siguiente"],
    rows, [q_next[0], q_next[1], q_next[2]],
    "Mapa de Lorenz (sigma=10, rho=28, beta=8/3), RK4, horizonte 0.1. Regresion multi-output (3 salidas) de un paso.",
)

# =====================================================================
# 10. DECAIMIENTO CON RUIDO (regresión con incertidumbre irreducible)
#     Cantidad final = inicial*exp(-K*t) + eps, eps ~ N(0, 3)
#     Ground truth del query = valor ESPERADO (sin ruido).
# =====================================================================
rng = np.random.default_rng(110)
SIGMA_NOISE = 3.0
rows = []
for _ in range(N_ROWS):
    A0 = rng.uniform(50, 500); K = rng.uniform(0.05, 0.5); t = rng.uniform(1, 20)
    val = A0*np.exp(-K*t) + rng.normal(0, SIGMA_NOISE)
    rows.append([A0, K, t, val])
qA0 = sample_interior(rng, 50, 500); qK = sample_interior(rng, 0.05, 0.5); qt = sample_interior(rng, 1, 20)
truth10 = qA0*np.exp(-qK*qt)   # esperado, SIN ruido
save_problem(
    10, "decaimiento_ruido",
    ["Cantidad final"],
    ["Cantidad inicial", "Factor K", "Tiempo"],
    [qA0, qK, qt],
    ["Cantidad inicial", "Factor K", "Tiempo", "Cantidad final"],
    rows, [truth10],
    "y = A0*exp(-K*t) + N(0,3). Ground truth = valor esperado sin ruido. Suelo de error teorico MAE ~ 2.39 (=sigma*sqrt(2/pi)).",
)

# =====================================================================
# Fichero de soluciones (ground truth para el harness de evaluación)
# =====================================================================
with open(os.path.join(OUT_DIR, "soluciones.txt"), "w", encoding="utf-8") as f:
    f.write("GROUND TRUTH DEL DATASET (queries excluidos de los CSV - regla R1)\n")
    f.write("Redondeo a 6 cifras significativas (R2). Queries en interpolacion (R3).\n")
    f.write("=" * 78 + "\n\n")
    for s in soluciones:
        f.write(s["archivo"] + "\n")
        f.write("  Query   : " + ", ".join("%s=%s" % (k, v) for k, v in s["inputs"].items()) + "\n")
        f.write("  Correcto: " + ", ".join("%s=%s" % (k, v) for k, v in s["outputs_correctos"].items()) + "\n")
        f.write("  Nota    : " + s["nota"] + "\n\n")

print("\nHecho. Archivos en", OUT_DIR)
