venv:
	python3 -m venv .venv

activate:
	source .venv/bin/activate

deactivate:
	deactivate

depinstall:
	pip3 install -e '.[visualization,dev]'

