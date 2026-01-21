bootstrap:
	python -m pip install -U pip
	python -m pip install -r requirements.txt
	python -m pip install -e .

run:
	streamlit run app.py

test:
	pytest -q

format:
	ruff check . --fix
	ruff format .
