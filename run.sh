rm -r -f lightning_logs
# CUDA_LAUNCH_BLOCKING=1 python umigame/experiments/run_text_classification.py
# CUDA_LAUNCH_BLOCKING=1 python umigame/experiments/run_textlstm.py
CUDA_LAUNCH_BLOCKING=1 python umigame/experiments/run_fasttext.py