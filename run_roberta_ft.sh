# Setting the pruning type to be None for standard fine-tuning.
TASK=$1
PRUNED_MODEL_PATH=$2
LEARNING_RATE=2e-5
SUFFIX=ft
EX_CATE=CoFi_RoBERTa

bash scripts/run_FT_roberta.sh $TASK $SUFFIX $EX_CATE $PRUNED_MODEL_PATH $LEARNING_RATE