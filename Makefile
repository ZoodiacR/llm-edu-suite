.PHONY: setup test notebooks run-notebooks smoke mini-llm mcp-server
setup:          ## create .venv and install everything (Linux/macOS; on Windows run setup_env.ps1)
	bash setup_env.sh
test:           ## unit + behavioural tests (~15 s)
	pytest -q
notebooks:      ## regenerate every .ipynb from tools/build_notebooks_part*.py (clears outputs)
	python tools/build_notebooks_part1.py && python tools/build_notebooks_part2.py && python tools/build_notebooks_part3.py
run-notebooks:  ## execute every notebook in place (~40 min on CPU)
	python tools/run_notebooks.py
smoke:          ## 30-second pre-training smoke test
	python scripts/train.py --steps 100 --d-model 64 --n-layers 2 --out runs/smoke --eval-every 50
mini-llm:       ## train the self-contained mini GPT (notebook 00 as a script)
	python scripts/mini_llm.py
mcp-server:     ## start the demo MCP server on stdio
	python scripts/mcp_server.py
