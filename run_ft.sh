# Setting the pruning type to be None for standard fine-tuning.
TASK=$1
PRUNED_MODEL_PATH=$2
LEARNING_RATE=3e-5
SUFFIX=ft
EX_CATE=CoFi

bash scripts/run_FT.sh $TASK $SUFFIX $EX_CATE $PRUNED_MODEL_PATH $LEARNING_RATE