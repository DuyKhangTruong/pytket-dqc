import csv
import os
import json
import math
import time
import argparse
import traceback
from pytket.qasm import circuit_from_qasm
from pytket.passes import DecomposeMultiQubitsCX
from pytket.circuit.display import view_browser
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.distributors import CoverEmbeddingSteinerDetached
from pytket_dqc.utils import DQCPass, max_ancillas_in_use

# Adjust these import paths based on where your custom refiners are located!
from pytket_dqc.refiners.intertwined_d_type_merge import IntertwinedDTypeMerge
from pytket_dqc.refiners.neighbouring_d_type_merge import NeighbouringDTypeMerge


# ---------------------------------------------------------------------------
# Path configuration  — edit these to match your local setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CIRCUITS_DIR   = os.path.join(_SCRIPT_DIR, "circuits")   # benchmark inputs
RESULTS_DIR    = os.path.join(_SCRIPT_DIR, "results")    # benchmark outputs

PARTITION_DIR  = os.path.join(CIRCUITS_DIR, "partitions")

# Benchmark suites: which folders to read the pure / graysynth circuits from.
#   default       the named benchmark circuits in CIRCUITS below
#   cnotdihedral  every .qasm file in circuits/qasm_cnotdihedral (copied there
#                 by export_cnotdihedral.sh in the feynman repo)
SUITES = {
    "default": {
        "qasm_dir":      os.path.join(CIRCUITS_DIR, "qasm"),
        "graysynth_dir": os.path.join(CIRCUITS_DIR, "graysynth_qasm"),
    },
    "cnotdihedral": {
        "qasm_dir":      os.path.join(CIRCUITS_DIR, "qasm_cnotdihedral"),
        "graysynth_dir": os.path.join(CIRCUITS_DIR, "graysynth_qasm_cnotdihedral"),
    },
}

# Circuits used by the "default" suite
CIRCUITS = [
    "adder_8",
    "barenco_tof_3",
    "barenco_tof_4",
    "barenco_tof_5",
    "barenco_tof_10",
    "csla_mux_3",
    "csum_mux_9",
    "gf2^4_mult",
    "gf2^5_mult",
    "gf2^6_mult",
    "gf2^7_mult",
    "gf2^8_mult",
    "gf2^9_mult",
    "gf2^10_mult",
    # "gf2^16_mult",
    "grover_5",
    "ham15-high",
    "ham15-low",
    "ham15-med",
    "hwb6",
    # "hwb8",
    # "hwb10",
    "mod_adder_1024",
    # "mod_adder_1048576",
    "mod_mult_55",
    "mod_red_21",
    "mod5_4",
    "qcla_adder_10",
    "qcla_com_7",
    "qcla_mod_7",
    "rc_adder_6",
    "tof_3",
    "tof_4",
    "tof_5",
    "tof_10",
    "vbe_adder_3",
    # "testing_circuit"
]

TOPOLOGIES = ("linear", "ring", "star", "grid", "full")

_TOPOLOGY_LABELS = {
    "linear": "Linear chain  (0--1--...--n-1)",
    "ring":   "Ring          (0--1--...--n-1--0)",
    "star":   "Star          (all servers connect to hub 0)",
    "grid":   "2-D Grid mesh (sqrt(n) x sqrt(n))",
    "full":   "Fully connected (complete graph, lower bound)",
}

ASSIGNMENT_METHODS = ("default", "kahypar")
_ASSIGNMENT_LABELS = {
    "default": "Default  (round-robin, evenly splits qubits across servers)",
    "kahypar": "KaHyPar  (reads .part file from partition directory)",
}


def circuits_for_suite(suite: str) -> list:
    """Circuit names (without .qasm) to run for the given suite."""
    if suite == "default":
        return CIRCUITS
    qasm_dir = SUITES[suite]["qasm_dir"]
    if not os.path.isdir(qasm_dir):
        return []
    # zero-padded names (cnotdihedral_01_..., _02_, ...) sort in generation order
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(qasm_dir) if f.endswith(".qasm")
    )


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------

def _describe_error(e: Exception) -> str:
    """One-line summary: exception type, message, and where it was raised.

    For a bare `assert` (empty message) the failing source line is shown instead,
    so errors are never just "Error:".
    """
    frames = traceback.extract_tb(e.__traceback__)
    msg = str(e).strip()
    where = ""
    if frames:
        last = frames[-1]
        where = f"  [at {os.path.basename(last.filename)}:{last.lineno} in {last.name}]"
        if not msg and last.line:
            msg = f"failed: {last.line.strip()}"
    return f"{type(e).__name__}: {msg or '(no message)'}{where}"


def _log_error(error_log: str, circuit: str, label: str, e: Exception) -> None:
    """Append the full traceback for one failure to the error log."""
    with open(error_log, "a") as f:
        f.write(f"=== {circuit} [{label}] ===\n")
        f.write("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        f.write("\n")


# ---------------------------------------------------------------------------
# Qubit assignment helpers
# ---------------------------------------------------------------------------

def _assign_qubits(total_qubits: int, n_partitions: int) -> dict:
    base, remainder = divmod(total_qubits, n_partitions)
    server_qubits = {}
    cursor = 0
    for sid in range(n_partitions):
        count = base + (1 if sid >= n_partitions - remainder else 0)
        server_qubits[sid] = list(range(cursor, cursor + count))
        cursor += count
    return server_qubits

def _kahypar_assign_qubits(filepath: str, total_qubits: int, n_partitions: int) -> dict:
    server_qubits = {i: [] for i in range(n_partitions)}

    if not os.path.exists(filepath):
        print(
            f"Warning: partition file not found at '{filepath}'. "
            "Falling back to default qubit assignment."
        )
        return _assign_qubits(total_qubits, n_partitions)

    with open(filepath, "r") as f:
        lines = f.read().splitlines()

    for q_idx in range(total_qubits):
        if q_idx < len(lines):
            part_id = int(lines[q_idx].strip())
            safe_part_id = part_id % n_partitions
            server_qubits[safe_part_id].append(q_idx)

    return server_qubits

def _resolve_assignment(
    method: str,
    part_file: str,
    total_qubits: int,
    n_partitions: int,
) -> dict:
    if method == "kahypar":
        return _kahypar_assign_qubits(part_file, total_qubits, n_partitions)
    return _assign_qubits(total_qubits, n_partitions)


# ---------------------------------------------------------------------------
# Topology builders
# ---------------------------------------------------------------------------

def _network_linear(total_qubits: int, n: int, method: str = "default", part_file: str = None) -> NISQNetwork:
    edges = [[i, i + 1] for i in range(n - 1)] or [[0, 0]]
    assignment = _resolve_assignment(method, part_file, total_qubits, n)
    return NISQNetwork(edges, assignment)

def _network_ring(total_qubits: int, n: int, method: str = "default", part_file: str = None) -> NISQNetwork:
    if n == 2:
        return _network_linear(total_qubits, n, method, part_file)
    edges = [[i, (i + 1) % n] for i in range(n)]
    assignment = _resolve_assignment(method, part_file, total_qubits, n)
    return NISQNetwork(edges, assignment)

def _network_star(total_qubits: int, n: int, method: str = "default", part_file: str = None) -> NISQNetwork:
    edges = [[0, i] for i in range(1, n)] or [[0, 0]]
    assignment = _resolve_assignment(method, part_file, total_qubits, n)
    return NISQNetwork(edges, assignment)

def _network_grid(total_qubits: int, n: int, method: str = "default", part_file: str = None) -> NISQNetwork:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    edges = []
    for r in range(rows):
        for c in range(cols):
            sid = r * cols + c
            if sid >= n:
                break
            right = sid + 1
            if c + 1 < cols and right < n:
                edges.append([sid, right])
            below = sid + cols
            if r + 1 < rows and below < n:
                edges.append([sid, below])
    if not edges:
        edges = [[0, 0]]
    assignment = _resolve_assignment(method, part_file, total_qubits, n)
    return NISQNetwork(edges, assignment)

def _network_full(total_qubits: int, n: int, method: str = "default", part_file: str = None) -> NISQNetwork:
    edges = [[i, j] for i in range(n) for j in range(i + 1, n)] or [[0, 0]]
    assignment = _resolve_assignment(method, part_file, total_qubits, n)
    return NISQNetwork(edges, assignment)

_TOPOLOGY_BUILDERS = {
    "linear": _network_linear,
    "ring":   _network_ring,
    "star":   _network_star,
    "grid":   _network_grid,
    "full":   _network_full,
}

def build_network(
    topology: str,
    total_qubits: int,
    n_partitions: int,
    assignment_method: str = "default",
    part_file: str = None,
) -> NISQNetwork:
    n = min(n_partitions, max(1, total_qubits))
    return _TOPOLOGY_BUILDERS[topology](total_qubits, n, assignment_method, part_file)


# ---------------------------------------------------------------------------
# Per-circuit processing
# ---------------------------------------------------------------------------

def process_circuit(
    filepath: str,
    distributor: CoverEmbeddingSteinerDetached,
    hypergraph_dir: str,
    distributed_dir: str,
    label: str,
    n_partitions: int,
    topology: str,
    assignment_method: str = "default",
    part_dir: str = None,
):
    tk_circuit = circuit_from_qasm(filepath)
    circuit_name = os.path.splitext(os.path.basename(filepath))[0]

    part_file = None
    if assignment_method == "kahypar" and part_dir:
        candidate = os.path.join(part_dir, f"{circuit_name}.part")
        if os.path.exists(candidate):
            part_file = candidate

    if tk_circuit.n_qubits < n_partitions:
        n_partitions = tk_circuit.n_qubits
        
    network = build_network(
        topology, tk_circuit.n_qubits, n_partitions, assignment_method, part_file
    )

    DecomposeMultiQubitsCX().apply(tk_circuit)
    DQCPass().apply(tk_circuit)

    IntertwinedDTypeMerge.total_blocked_merges = 0
    NeighbouringDTypeMerge.total_blocked_merges = 0

    # --- Timing starts: just before distribute() ---
    t_start = time.perf_counter()
    distribution = distributor.distribute(tk_circuit, network,seed=0)

    ebit_count = distribution.cost()

    # Save Kahypar Hypergraph (.hgr)
    hyp_path = os.path.join(hypergraph_dir, f"{circuit_name}_{label}_hypergraph.hgr")
    indices, vertices = distribution.circuit.kahypar_hyperedges()
    with open(hyp_path, "w") as f:
        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i+1]
            edge_qubits = vertices[start:end]
            line = " ".join(map(str, edge_qubits))
            f.write(line + "\n")

    # Save distributed circuit JSON
    dist_circ = distribution.to_pytket_circuit()
    # --- Timing ends: right after to_pytket_circuit() ---
    compile_time = time.perf_counter() - t_start

    # Peak number of link qubits (ancillas) alive at once, across all servers.
    # Computed after the timer stops so it doesn't affect the reported time.
    max_ancillas = max_ancillas_in_use(dist_circ)

    # view_browser(dist_circ)
    dist_path = os.path.join(distributed_dir, f"{circuit_name}_{label}_distributed.json")
    with open(dist_path, "w") as f:
        json.dump(dist_circ.to_dict(), f, indent=2)

    return ebit_count, max_ancillas, compile_time


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_and_export(
    output_csv: str,
    hypergraph_dir: str,
    distributed_dir: str,
    n_partitions: int,
    topology: str,
    assignment_method: str = "default",
    suite: str = "default",
    error_log: str = "errors.log",
):
    qasm_dir      = SUITES[suite]["qasm_dir"]
    graysynth_dir = SUITES[suite]["graysynth_dir"]
    circuits      = circuits_for_suite(suite)

    if not circuits:
        print(f"No circuits found for suite '{suite}' in {qasm_dir}")
        return

    pure_hyp_dir = os.path.join(hypergraph_dir, "Pure Pytket")
    gray_hyp_dir = os.path.join(hypergraph_dir, "Pytket Graysynth")
    os.makedirs(pure_hyp_dir, exist_ok=True)
    os.makedirs(gray_hyp_dir, exist_ok=True)

    pure_dist_dir = os.path.join(distributed_dir, "Pure Pytket")
    gray_dist_dir = os.path.join(distributed_dir, "Pytket Graysynth")
    os.makedirs(pure_dist_dir, exist_ok=True)
    os.makedirs(gray_dist_dir, exist_ok=True)

    distributor = CoverEmbeddingSteinerDetached()
    results = []
    n_errors = 0

    # start a fresh error log for this run
    open(error_log, "w").close()

    # column width for circuit names in the console output
    w = max(15, max(len(c) for c in circuits))

    print(f"\n{'='*62}")
    print(f"  Suite      : {suite} ({len(circuits)} circuits)")
    print(f"  Partitions : {n_partitions}")
    print(f"  Topology   : {_TOPOLOGY_LABELS[topology]}")
    print(f"  Assignment : {_ASSIGNMENT_LABELS[assignment_method]}")
    print(f"  QASM dir   : {qasm_dir}")
    print(f"  Graysynth  : {graysynth_dir}")
    print(f"  Partitions : {PARTITION_DIR}")
    print(f"{'='*62}\n")

    for circuit in circuits:
        filename = circuit + ".qasm"
        pure_path = os.path.join(qasm_dir, filename)
        gray_path = os.path.join(graysynth_dir, filename)

        # --- Pure pytket ---
        p_time, p_anc = None, ""
        if os.path.isfile(pure_path):
            try:
                p_ebit, p_anc, p_time = process_circuit(
                    pure_path, distributor, pure_hyp_dir, pure_dist_dir,
                    "pure", n_partitions, topology, assignment_method, PARTITION_DIR,
                )
                print(f"[pure]      {circuit:<{w}}: {p_ebit} e-bits | Max ancillas: {p_anc} | Time: {p_time:.3f}s")
            except Exception as e:
                p_ebit = f"Error: {_describe_error(e)}"
                print(f"[pure]      {circuit:<{w}}: {p_ebit}")
                _log_error(error_log, circuit, "pure", e)
                n_errors += 1
        else:
            p_ebit = "File not found"
            print(f"[pure]      {circuit:<{w}}: {p_ebit}")

        # --- Graysynth ---
        g_time, g_anc = None, ""
        if os.path.isfile(gray_path):
            try:
                g_ebit, g_anc, g_time = process_circuit(
                    gray_path, distributor, gray_hyp_dir, gray_dist_dir,
                    "graysynth", n_partitions, topology, assignment_method, PARTITION_DIR,
                )
                print(f"[graysynth] {circuit:<{w}}: {g_ebit} e-bits | Max ancillas: {g_anc} | Time: {g_time:.3f}s")
            except Exception as e:
                g_ebit = f"Error: {_describe_error(e)}"
                print(f"[graysynth] {circuit:<{w}}: {g_ebit}")
                _log_error(error_log, circuit, "graysynth", e)
                n_errors += 1
        else:
            g_ebit = "File not found"
            print(f"[graysynth] {circuit:<{w}}: {g_ebit}")

        results.append((
            circuit,
            p_ebit, p_anc,
            g_ebit, g_anc,
            p_time, g_time,
        ))

    _write_csv(results, output_csv)
    print(f"\nResults saved to:        {output_csv}")
    print(f"Hypergraphs (.hgr) to:   {hypergraph_dir}/")
    print(f"Distributed circuits to: {distributed_dir}/")
    if n_errors:
        print(f"{n_errors} error(s); full tracebacks in: {error_log}")
    else:
        os.remove(error_log)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

CSV_HEADERS = [
    "Circuit",
    "Pure E-bits", "Pure Max Ancillas",
    "Gray E-bits", "Gray Max Ancillas",
    "Pure Time (s)", "Gray Time (s)",
]


def _write_csv(results, output_path: str):
    """Write one row per circuit. Run settings are in the file name."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
        for row in results:
            *values, p_time, g_time = row
            # times to 3 decimals; empty when that circuit failed or was missing
            times = ["" if t is None else f"{t:.3f}" for t in (p_time, g_time)]
            writer.writerow([*values, *times])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distribute QASM circuits and export ebit counts to CSV.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        choices=tuple(SUITES),
        default="default",
        help=(
            "Which benchmark set to run. Default: default.\n"
            "  default       circuits in CIRCUITS, from circuits/qasm/ and\n"
            "                circuits/graysynth_qasm/\n"
            "  cnotdihedral  every circuit in circuits/qasm_cnotdihedral/ and\n"
            "                circuits/graysynth_qasm_cnotdihedral/"
        ),
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=2,
        help="Number of QPU partitions / servers (must be > 1). Default: 2.",
    )
    parser.add_argument(
        "--topology",
        choices=TOPOLOGIES,
        default="linear",
        help=(
            "Server interconnect topology. Default: linear.\n"
            "  linear  chain 0--1--...--n-1\n"
            "  ring    linear + wrap-around edge n-1--0\n"
            "  star    all servers connect to hub server 0\n"
            "  grid    2-D mesh, rows x cols ~ sqrt(n) x sqrt(n)\n"
            "  full    complete graph, every pair connected (lower bound)"
        ),
    )
    parser.add_argument(
        "--assignment",
        choices=ASSIGNMENT_METHODS,
        default="default",
        help=(
            "Qubit-to-server assignment method. Default: default.\n"
            "  default  Round-robin: splits qubits evenly across servers\n"
            "  kahypar  Reads a KaHyPar .part file from the partition directory"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--hypergraph-dir",
        default=None,
        help="Directory for hypergraph .hgr files.",
    )
    parser.add_argument(
        "--distributed-dir",
        default=None,
        help="Directory for distributed circuit JSON files.",
    )
    args = parser.parse_args()

    if args.partitions < 2:
        parser.error("Partitions must be > 1.")

    n      = args.partitions
    topo   = args.topology
    assign = args.assignment
    suite  = args.suite
    tag    = f"{n}partitions_{topo}_{assign}"
    if suite != "default":
        tag += f"_{suite}"   # keep cnotdihedral results separate from the standard ones

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_csv      = args.output         or os.path.join(RESULTS_DIR, f"ebit_counts_{tag}.csv")
    hypergraph_dir  = args.hypergraph_dir or os.path.join(RESULTS_DIR, f"circuit_hypergraphs_{tag}")
    distributed_dir = args.distributed_dir or os.path.join(RESULTS_DIR, f"distributed_circuits_{tag}")
    error_log       = os.path.join(RESULTS_DIR, f"errors_{tag}.log")

    run_all_and_export(output_csv, hypergraph_dir, distributed_dir, n, topo, assign, suite, error_log)