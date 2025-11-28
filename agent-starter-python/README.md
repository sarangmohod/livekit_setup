# Setup
install uv - curl -LsSf https://astral.sh/uv/install.sh | sh

Activate virtual environment using command - source .venv/bin/activate 

add dependencies - uv add -r requirements.txt



# Run agent
uv run src/agent.py 2>&1 | grep -v "NNPACK"
or uv run src/agent.py