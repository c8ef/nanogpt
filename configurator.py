"""
Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open("configurator.py").read())
"""

import sys
from ast import literal_eval
from pathlib import Path

for arg in sys.argv[1:]:
    if "=" not in arg:
        # assume it's the name of a config file
        assert not arg.startswith("--")
        config_file = arg
        print(f"Overriding config with {config_file}")
        config_file_content = Path(config_file).read_text()
        print(config_file_content)
        exec(config_file_content)
    else:
        # assume it's a --key=value argument
        assert arg.startswith("--")
        key, val = arg.split("=")
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            assert type(attempt) is type(globals()[key])
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
