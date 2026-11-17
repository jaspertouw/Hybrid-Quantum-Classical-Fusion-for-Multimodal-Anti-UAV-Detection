#!/bin/bash
# Submit only after the user approves this aggregate workload and runtime budget.
set -euo pipefail
readonly SOURCE_ROOT="${THESIS_SOURCE_ROOT:?Set the reviewed isolated source snapshot}"
readonly WORK_ROOT="${THESIS_WORK_ROOT:?Set the reviewed immutable grid plan}"
readonly JOB="${SOURCE_ROOT}/scripts/successor/fusion_grid.job"
test -f "${JOB}"
test -f "${WORK_ROOT}/PLAN.json"
test ! -e "${WORK_ROOT}/submission.txt"
python3 -c 'import json,os; p=json.load(open(os.environ["THESIS_WORK_ROOT"]+"/PLAN.json")); assert p["plan"]["mode"] == "grid" and len(p["plan"]["cases"]) == 20'
prepare=$(sbatch --parsable --time=00:10:00 --export=ALL,FUSION_STAGE=prepare "${JOB}")
printf 'prepare=%s\n' "${prepare}" >> "${WORK_ROOT}/submission.txt"
workers=$(sbatch --parsable --dependency="afterok:${prepare}" --array=0-19%2 --export=ALL,FUSION_STAGE=worker "${JOB}")
printf 'workers=%s\n' "${workers}" >> "${WORK_ROOT}/submission.txt"
aggregate=$(sbatch --parsable --time=00:10:00 --dependency="afterok:${workers}" --export=ALL,FUSION_STAGE=aggregate "${JOB}")
printf 'aggregate=%s\n' "${aggregate}" >> "${WORK_ROOT}/submission.txt"
cat "${WORK_ROOT}/submission.txt"
