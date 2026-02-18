yaml="configs/train/$1.yaml"
exp_name="VideoVAEPlus_$1"

n_HOST=1
elastic=1
GPUName="A"
current_time=$(date +%Y%m%d%H%M%S)

out_dir_name="${exp_name}_${n_HOST}nodes_e${elastic}_${GPUName}_$current_time"
res_root="./debug"

mkdir -p $res_root/$out_dir_name

torchrun \
--nproc_per_node=1 --nnodes=1 --master_port=16666 \
train.py \
--base $yaml \
-t --devices 0, \
lightning.trainer.num_nodes=1 \
--name ${out_dir_name} \
--logdir $res_root \
--auto_resume True \