TASK=SST2
SUFFIX=sparsity0.60_nodistill
EX_CATE=CoFi
PRUNING_TYPE=structured_heads+structured_mlp+hidden+layer
SPARSITY=0.60
DISTILL_LAYER_LOSS_ALPHA=0.9
DISTILL_CE_LOSS_ALPHA=0.1
LAYER_DISTILL_VERSION=4
SPARSITY_EPSILON=0.01
DISTILLATION_PATH=/data0/bowen/minus/bert_base_sst2_best_model

bash scripts/run_CoFi_nodistill.sh $TASK $SUFFIX $EX_CATE $PRUNING_TYPE $SPARSITY