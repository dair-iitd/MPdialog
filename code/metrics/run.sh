trap 'kill 0' SIGINT; # kill all child processes on Ctrl-C

CUDA_DEVICES=(1 2)
tok_dir="/home/user/MultiModal/Frozen/models/base/lm"
pred_file="ctx_resp_preds.pkl"
metric_file="eval_metrics.csv"
pred_dirs=(
  # "/home/user/MultiModal/Frozen/models/comic/linear2_clip_pgpt2_unfrozen_2eos_1/test/"
  "/home/user/MultiModal/Frozen/models/comic/linear2_clip_pgpt2_unfrozen_2eos_1/out_domain_all/"
  # "/home/user/MultiModal/Frozen/models/lm/dialogpt_comic/test/"
  # "/home/user/MultiModal/Frozen/models/lm/dialogpt_comic/out_domain_all/"
  # "/home/user/MultiModal/Frozen/models/lm/personaGPTcomic-p-double-eos/test/"
  # "/home/user/MultiModal/Frozen/models/lm/personaGPTcomic-p-double-eos/out_domain_all/"
  # "/home/user/EDGE-exemplars/runs/comic_in/34k_1/34k_1"
  # "/home/user/EDGE-exemplars/runs/comic_in/34k_1/20k_out_1/"
  # "/home/user/BoB/test_results/"
  # "/home/user/BoB/outdomain_results/"
)

function make_command {
  local _dir=$1
  echo "python -u main.py --input-df $_dir/$pred_file --output $_dir/$metric_file --tokenizer-dir $tok_dir"
  # echo "echo $_dir && sleep 1"
}

function wait_for_pid {
  local _pid=$1
  while [ -e /proc/$_pid ]; do
    sleep 0.1;
  done;
}

function test_pid {
  pid=
  for i in {0..2}; do
    if [ -z $pid ] 
    then
      command='echo "waiting for no one" && sleep 5 && echo "5s done" &'
    else
      command='echo "waiting for $pid" && wait_for_pid $pid && sleep 2 && echo "2s done" &'
    fi;
    eval $command
    pid=$!
    
  done;
  wait $pid
  echo "Test completed!"
}
input_df_dirs=()
for c in ${pred_dirs[@]}; do
  subdirs=$(find $c -mindepth 1 -type d)
  for x in ${subdirs[@]}; do
    input_df_dirs+=($x)
  done;
done;

n=${#input_df_dirs[@]}
num_devices=${#CUDA_DEVICES[@]}
device_barrier=$((n/num_devices + 1))
pid=
pids_to_wait=()

for i in ${!input_df_dirs[@]}; do
  
  command=$(make_command ${input_df_dirs[i]})
  command="CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[$((i / device_barrier))]} $command"

  if [ $((i % device_barrier)) -eq 0 ]
  then
    pids_to_wait+=($pid)
    unset pid
  fi;
  
  
  if [ -z $pid ]
  then
    command="echo 'waiting for no one' && $command"
  else
    command="echo 'waiting for $pid' && wait_for_pid $pid && $command"
  fi;

  command="$command &"
  echo "*** Executing: "$command
  eval $command
  pid=$!
  
done;

pids_to_wait+=($pid)

for pid_num in ${!pids_to_wait[@]}; do
  wait ${pids_to_wait[pid_num]}
  echo "Done with ${CUDA_DEVICES[pid_num]}"
done;



