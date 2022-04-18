#! /bin/sh

CONFIG_FILE=$1
source ${CONFIG_FILE}

python -u evaluate.py --dataset_name ${DATASET_NAME} --metadata_root ${METADATA_ROOT} --data_root ${DATA_ROOT} --wsol_method ${WSOL_METHOD} --vit_type ${VIT_TYPE} --adl_layer ${pADL_LAYER} --eval_method ${ATTENTION_GENERATION_METHOD} --experiment_name  ${EXPERIMENT_NAME} --ckpt_name ${CHECKPOINT_NAME} --batch_size ${BATCH_SIZE} --workers ${WORKERS} --scoremap_threshold ${SCOREMAP_THRESHOLD} --iou_threshold_list 30 50 70 --evaluate_mode True

