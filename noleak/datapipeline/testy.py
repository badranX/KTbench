
from pathlib import Path
import yaml


config = [str(p/"config.yaml") for p in Path(__file__).parents if p.name == "ktbench"][0]
#config = yaml.safe_load(open(config,'r'))
#print(config.keys())

import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append(config)
print(config)
from noleak import config
print(config.DATASETS_FOLDER)
print(config.__name__)

