#!/bin/bash

if [ $# -eq 0 ]; then
    echo "use: $0 <modelName>"
    echo "example: $0 LSTM"
    echo "or: $0 GRU"
    exit 1
fi

MODEL_NAME=$1


echo "use model: $MODEL_NAME"

nohup python search.py --model "$MODEL_NAME" \
    --past_features wd_deg SWDR_W max_wv wv_m rho_g max_PAR VPdef_mbar PAR_ol VPmax_mbar rh Tpot_K \
    > "${MODEL_NAME}_causalFeature.log" 2>&1 &

echo "logFile: ${MODEL_NAME}_NoMacro.log"
echo "cheak the log: tail -f ${MODEL_NAME}_causalFeature.log"
