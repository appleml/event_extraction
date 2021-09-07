"""
Helper functions.
"""

import os
import json
import data_prepare.file as file
import utils.constant as constant

### IO
def check_dir(d):
    if not os.path.exists(d):
        print("Directory {} does not exist. Exit.".format(d))
        exit(1)

def check_files(files):
    for f in files:
        if f is not None and not os.path.exists(f):
            print("File {} does not exist. Exit.".format(f))
            exit(1)

def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)

def save_config(config, path, verbose=True):
    with open(path, 'w') as outfile:
        json.dump(config, outfile, indent=2)
    if verbose:
        print("Config saved to file {}".format(path))
    return config

def load_config(path, verbose=True):
    with open(path) as f:
        config = json.load(f)
    if verbose:
        print("Config loaded from file {}".format(path))
    return config

def print_config(config):
    info = "Running with the following configs:\n"
    for k,v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    print("\n" + info + "\n")
    return

class FileLogger(object):
    """
    A file logger that opens the file periodically and write to it.
    """
    def __init__(self, filename, header=None):
        self.filename = filename
        if os.path.exists(filename):
            # remove the old file
            os.remove(filename)
        if header is not None:
            with open(filename, 'w') as out:
                print(header, file=out)
    
    def log(self, message):
        with open(self.filename, 'a') as out:
            print(message, file=out)

## 对gold_events进行处理
def process_gold_events(file_name, gold_events):
    simp_events = list()
    bind_events = list()
    pmod_events = list()
    regu_events = list()
    for event in gold_events:
        event_type = event.event_type
        if event_type in constant.SIMP_TYPE:
            simp_events.append(event)
        elif event_type in constant.BIND_TYPE:
            bind_events.append(event)
        elif event_type in constant.PMOD_TYPE:
            pmod_events.append(event)
        elif event_type in constant.REGU_TYPE:
            regu_events.append(event)
        else:
            print(event_type)
            print('have a problem!!!!')
    sent_events = file.sent_event(file_name)
    sent_events.add_simp(simp_events)
    sent_events.add_bind(bind_events)
    sent_events.add_pmod(pmod_events)
    sent_events.add_regu(regu_events)
    return sent_events