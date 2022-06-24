install:
	pip3 install --upgrade pip && pip3 install -r requirements.txt

format:
	black src/*.py

lint:
	pylint --disable=R,C src/train_gift_lightgbm.py

all: install lint format