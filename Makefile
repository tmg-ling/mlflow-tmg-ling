install:
	pip3 install --upgrade pip && pip3 install -r requirements.txt

format:
	black src/*.py

lint:
	pylint --disable=R,C lightgbm_gift/train.py tfrs_dcn_gift/train.py tfrs_two_tower_gift/train.py tfrs_listwise_ranking_gift/train.py

all: install lint format