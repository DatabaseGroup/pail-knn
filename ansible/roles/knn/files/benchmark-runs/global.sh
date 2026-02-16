export BINARY="./venv/bin/python3 run.py"
export REMOTE_SERVERS="cluster02,cluster04"
export LABEL_PREFIX="v11-"
export WORKING_DIR="/home/root/knn/experiments"

# Configuration variables
export CPU_CORES=8
export DATASET_DIR="../datasets"
export EXECUTABLE_DIR="../bin"
export EXECUTABLE="knn_stats"
export SAMPLE=10000
export PERFORM_STANDARD="true"
export PERFORM_SCALABILITY="true"

export DATASETS=( \
  # "arxiv-all-modes-labels" \
  "bms-pos-dedup-raw" \
  "dblp-all-modes-labels-raw" \
  # "dblpv14" \
  # "github-all-modes-labels" \
  "kosarak-dedup-raw" \
  "livejournal-userswithgroups-dedup-raw" \
  "lnonis1" \
  "movies-all-modes-labels" \
  "netflix" \
  "orkut-userswithgroups-dedup-raw" \
  "trackers" \
  "twitter" \
  # "twitter-all-modes-labels"
)

export K_VALUES=(5 10 20 40 80)
export CONCURRENCY="1"
export MIN_BATCH_SIZE=1
export MAX_BATCH_SIZE=1

export SCAL_CONCURRENCIES=(32 16 8 4 2)
export SCAL_K=20
export TOKEN_ID="${TOKEN_ID:-scal_tokens_$RANDOM}"

token_wrap() {
  local combined="$@"
  local concurrency="${combined%%|*}"
  local cmd="${combined#*|}"

  local token_dir="${WORKING_DIR}/locks"
  mkdir -p "$token_dir"
  local token_file="${token_dir}/.tokens.${TOKEN_ID}.count"
  local lock_file="${token_dir}/.tokens.${TOKEN_ID}.lock"

  # Initialize token count once (atomically) to CPU_CORES if missing.
  (
    exec 9>"$lock_file"
    flock 9
    if [[ ! -f "$token_file" ]]; then
      echo "${CPU_CORES:-1}" > "$token_file"
    fi
  )

  # Acquire tokens
  local acquired=0
  release_tokens() {
    if (( acquired == 1 )); then
      (
        exec 9>"$lock_file"
        flock 9
        local count
        count=$(cat "$token_file" 2>/dev/null || echo 0)
        ((count += concurrency))
        echo "$count" > "$token_file"
      )
      acquired=2
    fi
  }
  trap release_tokens EXIT

  while true; do
    (
      exec 9>"$lock_file"
      flock 9
      local count
      count=$(cat "$token_file" 2>/dev/null || echo 0)
      if (( count >= concurrency )); then
        ((count -= concurrency))
        echo "$count" > "$token_file"
        exit 0
      fi
      exit 1
    )
    [[ $? -eq 0 ]] && { acquired=1; break; }
    sleep 0.1
  done

  echo "Executing ${cmd}"
  eval "$cmd"
  local rc=$?

  # Release and clear trap
  release_tokens
  trap - EXIT
  return $rc
}
export -f token_wrap
