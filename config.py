from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ROOT = Path(__file__).resolve().parent
CORPUS_FILE = ROOT / "data" / "corpus.jsonl"
TRAIN_FILE = ROOT / "data" / "sft" / "train.jsonl"
VALID_FILE = ROOT / "data" / "sft" / "valid.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "qwen25_05b_pickup_lora"
RESULTS_DIR = ROOT / "results"
SYSTEM_PROMPT = (
    "你是一位幽默、尊重界線的土味情話高手。你會用親切、溫和、簡短的台灣繁體中文回答，"
    "依照前文延續雙關或反轉笑點；不騷擾、不貶低、不施壓，對方拒絕時就禮貌停止。"
)
SEED = 42

