#!/usr/bin/env bash
#
# End-to-end proving benchmark: the same fibonacci proofs on every GPU backend (CUDA, HIP, SYCL),
# so results from different machines and branches compare like for like.
#
# Full docs: sp1-gpu/scripts/bench-fib.md
#
# Quick examples (from the repo root):
#     sp1-gpu/scripts/bench-fib.sh target/release                      # one build
#     sp1-gpu/scripts/bench-fib.sh bench/main bench/feature            # A/B, alternating
#     sp1-gpu/scripts/bench-fib.sh --modes core --rounds 5 bench/a bench/b
#     sp1-gpu/scripts/bench-fib.sh --env-b "SP1_GPU_TRACE_PREFETCH_MIB=0" bench/a bench/a  # a setting

set -uo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] DIR_A [DIR_B]

Runs sp1-gpu-perf's \`node\` binary found in DIR_A (and DIR_B) on a fixed workload:

  core        local-fibonacci --param 2000000 (144,052,505 cycles), 4 proofs per process
  compressed  local-fibonacci --param 2000000, 2 proofs per process
  groth16     local-fibonacci --param 1000, 3 proofs per process

With two directories the builds alternate (A, B, A, B, ...) within each round, so drift in the
machine's state (clocks, temperature, background load) falls on both sides alike.

Options:
  --modes LIST     Comma-separated modes to run (default: core,compressed,groth16).
  --rounds N       Processes per side and mode (default: 3 for core and compressed, 2 for groth16).
  --out DIR        Where the logs go (default: bench-fib-<UTC timestamp>/).
  --timeout SECS   Per-process timeout (default: 1800).
  --env "K=V ..."  Extra environment for both sides, e.g. --env "SP1_GPU_ALLOC_CACHE_MB=16384".
  --env-b "K=V ..."
                   Extra environment for side B only. With the same directory twice, this
                   compares a setting rather than two builds.
  -h, --help       This help.

Each process's log is kept in --out. \`node\` verifies every proof and exits non-zero if one
fails. A shard whose proof fails verification is retried, which hides it from the exit code, so
the script also counts "verification failed" lines (the "bad" column). It should be 0.
EOF
}

MODES="core,compressed,groth16"
ROUNDS=""
OUT=""
TIMEOUT=1800
EXTRA_ENV=""
EXTRA_ENV_B=""
DIRS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --modes) MODES="$2"; shift 2 ;;
        --rounds) ROUNDS="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --env) EXTRA_ENV="$2"; shift 2 ;;
        --env-b) EXTRA_ENV_B="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) DIRS+=("$1"); shift ;;
    esac
done
if [ ${#DIRS[@]} -lt 1 ] || [ ${#DIRS[@]} -gt 2 ]; then
    usage >&2
    exit 2
fi

# The node binary in a directory (node, or node.exe on Windows), as an absolute path.
node_in() {
    local dir
    dir="$(cd "$1" 2>/dev/null && pwd)" || { echo "no such directory: $1" >&2; return 1; }
    if [ -x "$dir/node" ]; then echo "$dir/node"
    elif [ -x "$dir/node.exe" ]; then echo "$dir/node.exe"
    else echo "no node binary in $1" >&2; return 1
    fi
}
SIDES=(A)
BIN_A="$(node_in "${DIRS[0]}")" || exit 2
if [ ${#DIRS[@]} -eq 2 ]; then
    SIDES=(A B)
    BIN_B="$(node_in "${DIRS[1]}")" || exit 2
fi

# The workload: param, proofs per process, default rounds.
workload() {
    case "$1" in
        core) echo "2000000 4 3" ;;
        compressed) echo "2000000 2 3" ;;
        groth16) echo "1000 3 2" ;;
        *) return 1 ;;
    esac
}
for mode in ${MODES//,/ }; do
    workload "$mode" >/dev/null || { echo "unknown mode: $mode" >&2; exit 2; }
done

# `node` runs from the repo root (rsp inputs and other relative paths resolve from there).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-bench-fib-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

export SP1_SKIP_PROGRAM_BUILD=true
export RUST_LOG="${RUST_LOG:-info}"

{
    echo "# bench-fib $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "A: $BIN_A"
    [ ${#SIDES[@]} -eq 2 ] && echo "B: $BIN_B"
    echo "host: $(uname -srm)"
    gpus=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null) &&
        echo "$gpus" | sed 's/^/gpu: /'
    env | grep -E '^(SP1_|CUDA_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES|ONEAPI_DEVICE_SELECTOR)' | sort |
        sed 's/^/env: /'
    [ -n "$EXTRA_ENV" ] && echo "extra env: $EXTRA_ENV"
    [ -n "$EXTRA_ENV_B" ] && echo "extra env, B only: $EXTRA_ENV_B"
} | tee "$OUT/summary.txt"

# One process: prints "<rc> <bad> <time> <time> ...".
run_one() {
    local bin="$1" mode="$2" param="$3" n="$4" log="$5" side_env="$6"
    (cd "$REPO_ROOT" && env $EXTRA_ENV $side_env timeout "$TIMEOUT" "$bin" --program local-fibonacci \
        --param "$param" --mode "$mode" -n "$n") >"$log" 2>&1
    local rc=$?
    local bad times
    bad=$(grep -ac 'verification failed' "$log")
    times=$(grep -aoE 'proof time: [0-9.]+' "$log" | grep -oE '[0-9.]+$' | tr '\n' ' ')
    echo "$rc $bad $times"
}

for mode in ${MODES//,/ }; do
    read -r param n default_rounds <<<"$(workload "$mode")"
    rounds="${ROUNDS:-$default_rounds}"
    echo "## $mode: --param $param, $n proofs per process, $rounds rounds" | tee -a "$OUT/summary.txt"
    for r in $(seq "$rounds"); do
        # Alternate which side goes first each round.
        order=("${SIDES[@]}")
        if [ ${#SIDES[@]} -eq 2 ] && [ $((r % 2)) -eq 0 ]; then order=(B A); fi
        for side in "${order[@]}"; do
            bin="$BIN_A"; side_env=""
            if [ "$side" = B ]; then bin="$BIN_B"; side_env="$EXTRA_ENV_B"; fi
            log="$OUT/$mode-$side-$r.log"
            read -r rc bad times <<<"$(run_one "$bin" "$mode" "$param" "$n" "$log" "$side_env")"
            echo "$mode $side $r $rc $bad $times" >>"$OUT/results.txt"
            printf '%-10s round %d %s  rc=%s bad=%s  %s\n' "$mode" "$r" "$side" "$rc" "$bad" "$times" |
                tee -a "$OUT/summary.txt"
        done
    done
done

# Medians of the first proof and of the later (steady-state) proofs per side and mode, and B's
# change against A.
awk '
# The median of a space-separated list of numbers, or "" for an empty list.
function median(list,    n, a, i, j, t) {
    n = split(list, a, " ")
    if (n == 0) return ""
    for (i = 1; i <= n; i++) a[i] += 0
    for (i = 2; i <= n; i++) { t = a[i]; for (j = i - 1; j >= 1 && a[j] > t; j--) a[j + 1] = a[j]; a[j + 1] = t }
    return (n % 2) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2
}
function fmt(x) { return (x == "") ? "-" : sprintf("%.3f", x) }
function change(a, b) { return (a > 0 && b != "") ? sprintf("%+.1f%%", (b / a - 1) * 100) : "-" }
{
    key = $1 SUBSEP $2
    if (!($1 in modes)) { modes[$1] = 1; order[++nmodes] = $1 }
    if ($2 == "B") have_b = 1
    if ($4 != 0) failed[key]++
    bad[key] += $5
    if (NF >= 6) first[key] = first[key] " " $6
    for (i = 7; i <= NF; i++) steady[key] = steady[key] " " $i
}
END {
    printf "\n%-10s %-4s %10s %10s %6s %6s\n", "mode", "side", "first(s)", "steady(s)", "failed", "bad"
    nsides = have_b ? 2 : 1
    for (i = 1; i <= nmodes; i++) for (k = 1; k <= nsides; k++) {
        m = order[i]; s = (k == 1) ? "A" : "B"; key = m SUBSEP s
        f[key] = median(first[key]); st[key] = median(steady[key])
        printf "%-10s %-4s %10s %10s %6d %6d\n", m, s, fmt(f[key]), fmt(st[key]), failed[key], bad[key]
    }
    if (have_b) {
        printf "\n%-10s %12s %12s\n", "mode", "first B/A", "steady B/A"
        for (i = 1; i <= nmodes; i++) {
            m = order[i]
            printf "%-10s %12s %12s\n", m, change(f[m SUBSEP "A"], f[m SUBSEP "B"]),
                change(st[m SUBSEP "A"], st[m SUBSEP "B"])
        }
    }
}' "$OUT/results.txt" | tee -a "$OUT/summary.txt"

echo
echo "logs and summary: $OUT"
