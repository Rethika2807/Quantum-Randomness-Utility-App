# qrng_app_pro.py
# Works with: Python 3.13, qiskit-terra >= 0.46, qiskit-aer >= 0.14
# Optional: qiskit-ibm-runtime (for real hardware/cloud backends)

import math
from collections import Counter
from typing import Optional

import streamlit as st
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import matplotlib.pyplot as plt

# Optional IBM Runtime support (safe fallback if not installed)
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    _IBM_AVAILABLE = True
except Exception:
    _IBM_AVAILABLE = False

# --------------------------
# Helpers: QRNG Core
# --------------------------
@st.cache_resource(show_spinner=False)
def get_aer_backend():
    return Aer.get_backend("qasm_simulator")

@st.cache_resource(show_spinner=False)
def get_ibm_backend(backend_name: str = "ibmq_qasm_simulator"):
    """Return an IBM Runtime backend if available & configured.
    Requires: pip install qiskit-ibm-runtime, and user has a saved account/token.
    """
    if not _IBM_AVAILABLE:
        raise RuntimeError("qiskit-ibm-runtime not installed")
    try:
        svc = QiskitRuntimeService()  # assumes user configured account earlier
        backend = svc.backend(backend_name)
        return backend
    except Exception as e:
        raise RuntimeError(f"IBM backend not available: {e}")


def run_circuit_and_get_counts(qc: QuantumCircuit, shots: int = 1, backend_choice: str = "Aer", ibm_backend_name: str = "ibmq_qasm_simulator") -> Counter:
    """Execute circuit and return counts dict as Counter.
    Uses backend.run() (modern API).
    """
    if backend_choice == "IBM Runtime":
        try:
            backend = get_ibm_backend(ibm_backend_name)
        except Exception:
            backend = get_aer_backend()  # silent fallback to Aer
    else:
        backend = get_aer_backend()

    job = backend.run(qc, shots=shots)
    res = job.result()
    counts = res.get_counts()
    # Qiskit may return a dict or a list (for multiple experiments); normalize
    if isinstance(counts, list):
        counts = counts[0]
    return Counter(counts)


def qrng_bits(n_bits: int, backend_choice: str = "Aer", ibm_backend_name: str = "ibmq_qasm_simulator") -> str:
    """Generate n_bits quantum random bits as a string like '0101'."""
    if n_bits <= 0:
        return ""
    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    qc.measure(range(n_bits), range(n_bits))
    counts = run_circuit_and_get_counts(qc, shots=1, backend_choice=backend_choice, ibm_backend_name=ibm_backend_name)
    bitstring = next(iter(counts))  # e.g., {'0101': 1}
    # Reverse for user-friendly MSB-left ordering
    return bitstring[::-1]


def qrng_counts_one_qubit(shots: int, backend_choice: str = "Aer", ibm_backend_name: str = "ibmq_qasm_simulator") -> Counter:
    if shots <= 0:
        return Counter()
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    return run_circuit_and_get_counts(qc, shots=shots, backend_choice=backend_choice, ibm_backend_name=ibm_backend_name)


def qrng_int(max_exclusive: int, backend_choice: str = "Aer", ibm_backend_name: str = "ibmq_qasm_simulator") -> int:
    """Uniform integer in [0, max_exclusive) via rejection sampling (no modulo bias)."""
    if max_exclusive <= 1:
        return 0
    bits_needed = math.ceil(math.log2(max_exclusive))
    while True:
        b = qrng_bits(bits_needed, backend_choice, ibm_backend_name)
        val = int(b, 2)
        if val < max_exclusive:
            return val


def qrng_choice(seq, backend_choice: str = "Aer", ibm_backend_name: str = "ibmq_qasm_simulator"):
    idx = qrng_int(len(seq), backend_choice, ibm_backend_name)
    return seq[idx]

# --------------------------
# Stats & Utilities
# --------------------------

def chunk(s: str, k: int) -> str:
    return " ".join(s[i:i+k] for i in range(0, len(s), k))


def to_hex_from_bits(bits: str) -> str:
    pad = (-len(bits)) % 4
    if pad:
        bits = bits + "0" * pad
    return hex(int(bits, 2))[2:].upper()


def monobit_frequency_test(bits: str):
    if not bits:
        return 0.0, 0.0
    n = len(bits)
    s_obs = sum(1 if b == "1" else -1 for b in bits)
    s = abs(s_obs) / math.sqrt(n)
    p = math.erfc(s / math.sqrt(2))
    return s, p


def runs_test(bits: str):
    if not bits:
        return 0, 0.0
    n = len(bits)
    pi = bits.count("1") / n
    if abs(pi - 0.5) > 0.495:
        return 0, 0.0
    runs = 1 + sum(bits[i] != bits[i-1] for i in range(1, n))
    expected = 2 * n * pi * (1 - pi) + 1
    var = 2 * n * (2 * pi * (1 - pi))**2
    if var == 0:
        return runs, 0.0
    z = (runs - expected) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return runs, p


def shannon_entropy(bits: str):
    if not bits:
        return 0.0
    n0 = bits.count("0")
    n1 = len(bits) - n0
    p0 = n0 / len(bits)
    p1 = n1 / len(bits)
    def h(p):
        return 0 if p in (0, 1) else -p * math.log2(p)
    return h(p0) + h(p1)


def chi_square_for_counts(c0: int, c1: int):
    total = c0 + c1
    if total == 0:
        return 0.0, 1.0
    expct = total / 2
    chi = (c0 - expct) ** 2 / expct + (c1 - expct) ** 2 / expct
    p_approx = math.exp(-chi / 2)
    return chi, p_approx


def autocorr_lag1(bits: str) -> float:
    """Lag-1 autocorrelation in {-1, +1} mapping. 0 means no linear correlation."""
    if len(bits) < 2:
        return 0.0
    xs = [1 if b == '1' else -1 for b in bits]
    n = len(xs) - 1
    num = sum(xs[i] * xs[i+1] for i in range(n))
    den = sum(x*x for x in xs[:-1])  # == n since x in {-1,1}
    return num / den if den else 0.0

# --------------------------
# UI Styling
# --------------------------
st.set_page_config(page_title="Quantum Randomness Utility (Pro)", page_icon="⚛", layout="wide")
st.markdown(
    """
    <style>
    .card { border-radius: 16px; padding: 16px 18px; border: 1px solid rgba(125,125,125,0.2); box-shadow: 0 2px 12px rgba(0,0,0,0.06);} 
    .big { font-size: 1.35rem; font-weight: 700; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; }
    .badge { display:inline-block; padding:3px 10px; border-radius:999px; background:rgba(0,0,0,0.06); font-size:0.85rem; margin-left:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚛ Quantum Randomness Utility — Pro Edition")
st.caption("Superposition → Measurement → *True* random bits. Built with Qiskit. Now with circuit visualization and quantum XOR demo.")

with st.sidebar:
    st.header("Controls")
    backend_choice = st.selectbox(
        "Backend",
        ["Aer", "IBM Runtime"],
        index=0,
        help="IBM Runtime requires qiskit-ibm-runtime installed and your account saved. Falls back to Aer if unavailable.",
    )
    ibm_backend_name = st.text_input("IBM backend name", value="ibmq_qasm_simulator")

    feature = st.radio(
        "Choose a feature",
        [
            "Generate Random Bits",
            "Coin Toss (live & multi)",
            "Dice Roller",
            "Lottery Generator",
            "Password Generator",
            "Encryption Key",
            "Quantum XOR Encrypt",
            "Quantum vs Classical",
            "Statistical Tests",
            "About",
        ],
    )

# --------------------------
# Feature: Generate Random Bits (+ Circuit Viz)
# --------------------------
if feature == "Generate Random Bits":
    c1, c2 = st.columns([1, 1])
    with c1:
        n = st.slider("Number of bits", min_value=4, max_value=1024, value=64, step=4)
        bits = qrng_bits(n, backend_choice, ibm_backend_name)
        st.markdown(
            f"<div class='card'><div class='big'>🎯 Quantum Random Bits</div><div class='mono'>{chunk(bits, 4)}</div></div>",
            unsafe_allow_html=True,
        )
        st.write(f"Length: *{len(bits)}* bits  |  0s: *{bits.count('0')}*  |  1s: *{bits.count('1')}*")
        st.download_button("📥 Download bits (.txt)", data=bits, file_name="qrng_bits.txt", mime="text/plain")
    with c2:
        lbls = ["0", "1"]
        counts = [bits.count("0"), bits.count("1")]
        fig, ax = plt.subplots()
        ax.bar(lbls, counts)
        ax.set_title("Bit Counts")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with st.expander("Show Quantum Circuit Diagram"):
        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        qc.measure(range(n), range(n))
        st.pyplot(qc.draw("mpl"))

# --------------------------
# Feature: Coin Toss
# --------------------------
elif feature == "Coin Toss (live & multi)":
    st.subheader("🪙 Quantum Coin Toss")
    colA, colB = st.columns([1, 1])
    with colA:
        single = qrng_bits(1, backend_choice, ibm_backend_name)
        face = "Heads 🎉" if single == "0" else "Tails 🎯"
        st.markdown(f"<div class='card big'>Result: <b>{face}</b></div>", unsafe_allow_html=True)

        shots = st.slider("Number of tosses (for chart)", 10, 5000, 200)
        counts = qrng_counts_one_qubit(shots, backend_choice, ibm_backend_name)
        c0, c1 = counts.get("0", 0), counts.get("1", 0)
        st.write(f"Totals → Heads: *{c0}, Tails: **{c1}* (n={shots})")
        chi, p = chi_square_for_counts(c0, c1)
        st.caption(f"Fairness check (χ²≈{chi:.3f}, p≈{p:.3f}). Higher p ≈ more fair.")
    with colB:
        fig, ax = plt.subplots()
        ax.bar(["Heads(0)", "Tails(1)"], [counts.get("0", 0), counts.get("1", 0)])
        ax.set_title("Coin Toss Distribution")
        ax.set_ylabel("Count")
        st.pyplot(fig)

# --------------------------
# Feature: Dice Roller
# --------------------------
elif feature == "Dice Roller":
    st.subheader("🎲 Quantum Dice")
    rolls = st.slider("Number of rolls", 1, 50, 6)
    results = [qrng_int(6, backend_choice, ibm_backend_name) + 1 for _ in range(rolls)]
    st.markdown(
        f"<div class='card'>Results: <span class='big'>{' • '.join(map(str, results))}</span></div>",
        unsafe_allow_html=True,
    )
    fig, ax = plt.subplots()
    ax.bar(range(1, 7), [results.count(i) for i in range(1, 7)])
    ax.set_title("Dice Outcome Distribution")
    ax.set_xlabel("Face")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# --------------------------
# Feature: Lottery Generator
# --------------------------
elif feature == "Lottery Generator":
    st.subheader("🎟 Quantum Lottery")
    max_n = st.number_input("Max number (inclusive)", min_value=10, max_value=100, value=60, step=1)
    picks = st.number_input("How many unique picks?", min_value=1, max_value=10, value=6, step=1)
    pool = set()
    while len(pool) < picks:
        val = qrng_int(int(max_n), backend_choice, ibm_backend_name) + 1
        pool.add(val)
    nums = sorted(pool)
    st.markdown(
        f"<div class='card big'>Your quantum picks: {', '.join(map(str, nums))}</div>",
        unsafe_allow_html=True,
    )
    st.download_button(
        "📥 Download picks (.txt)",
        data=", ".join(map(str, nums)),
        file_name="qrng_lottery.txt",
        mime="text/plain",
    )

# --------------------------
# Feature: Password Generator
# --------------------------
elif feature == "Password Generator":
    st.subheader("🔐 Quantum Passwords")
    use_upper = st.checkbox("A–Z", True)
    use_lower = st.checkbox("a–z", True)
    use_digits = st.checkbox("0–9", True)
    use_symbols = st.checkbox("Symbols (!@#$%^&*)", True)
    length = st.slider("Password length", 6, 32, 12)

    U = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if use_upper else ""
    L = "abcdefghijklmnopqrstuvwxyz" if use_lower else ""
    D = "0123456789" if use_digits else ""
    S = "!@#$%^&*" if use_symbols else ""
    charset = U + L + D + S
    if not charset:
        st.error("Select at least one character set.")
    else:
        pwd_chars = []
        required_sets = [s for s in [U, L, D, S] if s]
        for rs in required_sets:
            pwd_chars.append(qrng_choice(rs, backend_choice, ibm_backend_name))
        while len(pwd_chars) < length:
            pwd_chars.append(qrng_choice(charset, backend_choice, ibm_backend_name))
        # quantum shuffle via random index swaps
        for i in range(len(pwd_chars)):
            j = qrng_int(len(pwd_chars), backend_choice, ibm_backend_name)
            pwd_chars[i], pwd_chars[j] = pwd_chars[j], pwd_chars[i]
        password = "".join(pwd_chars[:length])
        st.markdown(
            f"<div class='card big'>Password: <span class='mono'>{password}</span></div>",
            unsafe_allow_html=True,
        )
        st.caption("Generated with quantum-chosen indices (no classical PRNG).")

# --------------------------
# Feature: Encryption Key
# --------------------------
elif feature == "Encryption Key":
    st.subheader("🔑 Quantum Key Generator")
    key_bits = st.selectbox("Key size (bits)", [32, 64, 128, 256, 512], index=3)
    bits = qrng_bits(key_bits, backend_choice, ibm_backend_name)
    grouped = chunk(bits, 4)
    as_hex = to_hex_from_bits(bits)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            f"<div class='card'><div class='big'>Binary</div><div class='mono'>{grouped}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='card'><div class='big'>Hex</div><div class='mono'>{chunk(as_hex, 4)}</div></div>",
            unsafe_allow_html=True,
        )
    colD1, colD2 = st.columns([1, 1])
    with colD1:
        st.download_button(
            "📥 Download (binary .txt)", data=bits, file_name=f"qrng_key_{key_bits}b.txt", mime="text/plain"
        )
    with colD2:
        st.download_button(
            "📥 Download (hex .txt)", data=as_hex, file_name=f"qrng_key_{key_bits}b_hex.txt", mime="text/plain"
        )

# --------------------------
# NEW Feature: Quantum XOR Encrypt (demo)
# --------------------------
elif feature == "Quantum XOR Encrypt":
    st.subheader("🔒 Quantum XOR Encryption Demo")
    msg = st.text_area("Enter message to encrypt", value="Hello, Quantum World!", help="Plain text will be XORed with a one-time quantum key of equal length.")
    if st.button("Generate key & Encrypt"):
        if not msg:
            st.warning("Please enter a message.")
        else:
            data = msg.encode("utf-8")
            key_bits = len(data) * 8
            bits = qrng_bits(key_bits, backend_choice, ibm_backend_name)
            key_bytes = int(bits, 2).to_bytes(len(data), byteorder="big")
            cipher = bytes(d ^ k for d, k in zip(data, key_bytes))
            st.markdown("*Quantum Key (hex):*")
            st.code(key_bytes.hex().upper())
            st.markdown("*Ciphertext (hex):*")
            st.code(cipher.hex().upper())
            st.download_button("📥 Download ciphertext (.bin)", data=cipher, file_name="cipher.bin")
            st.download_button("📥 Download key (.bin)", data=key_bytes, file_name="key.bin")

            with st.expander("Decrypt with the same key (verify)"):
                recovered = bytes(c ^ k for c, k in zip(cipher, key_bytes))
                try:
                    st.code(recovered.decode("utf-8"))
                except Exception:
                    st.code(str(recovered))
            st.caption("This illustrates one-time-pad style XOR using a quantum-generated key.")

# --------------------------
# Feature: Quantum vs Classical
# --------------------------
elif feature == "Quantum vs Classical":
    from secrets import randbits  # strong classical RNG for comparison

    st.subheader("📊 Quantum vs Classical RNG")
    n = st.slider("Number of bits", 16, 4096, 128, step=16)
    q_bits = qrng_bits(n, backend_choice, ibm_backend_name)
    c_bits = bin(randbits(n))[2:].zfill(n)

    st.markdown("#### Samples")
    st.code(f"Quantum : {chunk(q_bits, 4)}")
    st.code(f"Classical: {chunk(c_bits, 4)}")

    q0, q1 = q_bits.count("0"), q_bits.count("1")
    c0, c1 = c_bits.count("0"), c_bits.count("1")
    chi_q, p_q = chi_square_for_counts(q0, q1)
    chi_c, p_c = chi_square_for_counts(c0, c1)

    c1_, c2_ = st.columns([1, 1])
    with c1_:
        fig1, ax1 = plt.subplots()
        ax1.bar(["0", "1"], [q0, q1])
        ax1.set_title(f"Quantum counts (χ²≈{chi_q:.3f}, p≈{p_q:.3f})")
        st.pyplot(fig1)
    with c2_:
        fig2, ax2 = plt.subplots()
        ax2.bar(["0", "1"], [c0, c1])
        ax2.set_title(f"Classical counts (χ²≈{chi_c:.3f}, p≈{p_c:.3f})")
        st.pyplot(fig2)

# --------------------------
# Feature: Statistical Tests (+ Autocorrelation)
# --------------------------
elif feature == "Statistical Tests":
    st.subheader("🧪 Quick Statistical Checks (demo)")
    n = st.slider("Generate bits", 32, 8192, 512, step=32)
    bits = qrng_bits(n, backend_choice, ibm_backend_name)
    S, p_freq = monobit_frequency_test(bits)
    runs, p_runs = runs_test(bits)
    H = shannon_entropy(bits)
    rho1 = autocorr_lag1(bits)

    st.markdown(
        f"""
<div class='card'>
  <div class='big'>Results on {n} quantum bits</div>
  <ul>
    <li><b>Monobit Frequency</b>: S = {S:.3f}, p ≈ {p_freq:.3f}</li>
    <li><b>Runs Test</b>: runs = {runs}, p ≈ {p_runs:.3f}</li>
    <li><b>Shannon Entropy</b>: H ≈ {H:.3f} bits/bit (max = 1.0)</li>
    <li><b>Autocorrelation (lag 1)</b>: ρ₁ ≈ {rho1:.4f} (closer to 0 is better)</li>
  </ul>
  <div class='mono'>{chunk(bits[:256], 4)}{' ...' if len(bits)>256 else ''}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    csv = "index,bit\n" + "\n".join(f"{i},{b}" for i, b in enumerate(bits))
    st.download_button("📥 Download bits as CSV", data=csv, file_name="qrng_bits.csv", mime="text/csv")

# --------------------------
# About
# --------------------------
elif feature == "About":
    st.subheader("ℹ About this App")
    st.markdown(
        """
*What makes this different (judge view):*
- Multiple *applied features*: coin toss, dice, lottery, passwords, keys.
- *Quantum XOR encryption demo* using one-time key derived from QRNG.
- *Unbiased sampling* (rejection sampling) for dice/lottery — no modulo bias.
- *Circuit visualization* to show superposition → measurement clearly.
- *Fairness checks: χ², monobit, runs, entropy, **autocorrelation*.
- *Backend selector: Aer by default; optional **IBM Runtime* if configured.
- *Readable outputs* (grouped bits, hex) + *downloads* (txt/csv/bin).
- *Modern Qiskit API* (backend.run) — robust on latest versions.

*How it works (simple):*
Hadamard creates superposition → measurement yields truly random bits → mapped to features (ints, choices, keys, XOR crypto).

*Optional IBM setup:*
pip install qiskit-ibm-runtime then save your account once in Python:
python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum", token="<YOUR_API_TOKEN>")

Restart the app, choose *IBM Runtime*, and provide a backend name (e.g., ibmq_qasm_simulator or a real device if you have access).
"""
    )