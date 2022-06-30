install:
	pip3 install --upgrade pip && pip3 install -r requirements.txt

format:
	black src/*.py

lint:
	pylint --disable=R,C tfrs_dcm_gift/train.py

all: install lint format