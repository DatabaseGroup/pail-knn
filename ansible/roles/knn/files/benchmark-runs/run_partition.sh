generate_partition_args() {
    local dataset="$1"
    local k="$2"
    local sample="$3"
    local concurrency="$4"

    local dataset_file="${DATASET_DIR}/${dataset}"
    local fulllabel="${LABEL_PREFIX}${LABEL}"
    local arg_string="${EXECUTABLE_DIR}/${EXECUTABLE} -l ${fulllabel} -a ${ALGORITHM} -c ${concurrency} -f ${dataset_file} -k ${k} -m ${MODE} -s ${sample}"
    echo "${arg_string}"
}

# Standard Experiment
if [[ ${PERFORM_STANDARD} == "true" ]]; then
  ARGS=()
  for dataset in "${DATASETS[@]}"; do
      for k in "${K_VALUES[@]}"; do
          arg_string=$(generate_partition_args "${dataset}" "${k}" "${SAMPLE}" "${CONCURRENCY}")
          ARGS+=("${arg_string}")
      done
  done

  parallel -S "${REMOTE_SERVERS}" \
    --controlmaster \
    --progress \
    --workdir "${WORKING_DIR}" \
    -j "${CPU_CORES}" \
    --shuf \
    --colsep ' ' \
    "${BINARY} {}" ::: "${ARGS[@]}"
fi

# Scalability Experiment
if [[ ${PERFORM_SCALABILITY} == "true" ]]; then
  CONC_ARGS=()
  for concurrency in "${SCAL_CONCURRENCIES[@]}"; do
    sample=$((SAMPLE * concurrency))
    for dataset in "${DATASETS[@]}"; do
      arg_string=$(generate_partition_args "${dataset}" "${SCAL_K}" "${sample}" "${concurrency}")
      CONC_ARGS+=("$concurrency|$BINARY $arg_string")
    done
  done

  parallel --env token_wrap,CPU_CORES,WORKING_DIR,TOKEN_ID \
    -S "${REMOTE_SERVERS}" \
    --controlmaster \
    --progress \
    --workdir "${WORKING_DIR}" \
    -j "$((CPU_CORES / 4))" \
    --shuf \
    --colsep ' ' \
    "token_wrap {}" ::: "${CONC_ARGS[@]}"
fi