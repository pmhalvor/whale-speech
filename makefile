local-run: 
	bash scripts/kill_model_server.sh
	python3 src/create_table.py
	python3 src/model_server.py & python3 src/pipeline.py
	bash scripts/kill_model_server.sh

run-pipeline:
	python3 src/pipeline.py

model-server:
	python3 src/model_server.py

kill-model-server:
	bash scripts/kill_model_server.sh

create-table:
	python3 src/create_table.py