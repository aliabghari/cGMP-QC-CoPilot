.PHONY: test verify demo install lint

test:
	pytest

verify:
	python verify_hybrid.py

demo:
	streamlit run app.py

lint:
	ruff check .

install:
	pip install -r requirements.txt
