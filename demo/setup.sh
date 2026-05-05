sudo snap install ollama
ollama pull qwen2.5:3b
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt