python main.py \
  --model LSTMCausalAd \
  --data_path ./data/weather.csv \
  --target OT \
  --past_features wd_deg SWDR_W max_wv wv_m rho_g max_PAR VPdef_mbar PAR_ol VPmax_mbar rh Tpot_K \
  --forward_features month year \
  --sequence_length 96 \
  --step_forward 96 \
  --batch_size 1024 \
  --lr 0.01 \
  --epochs 50 \
  --fix_seed True

# python main.py --model {LSTM, LSTM_Attention, LSTMPostMarkAttCausalAd, TransformerModel, TimeMixer} 