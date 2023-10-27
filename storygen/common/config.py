# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Any
import yaml


def recursive_lowercase_keys(d):
    if type(d) is dict:
        new_d = {}
        for key in d:
            new_d[key.lower()] = recursive_lowercase_keys(d[key])
        return new_d
    else:
        return d


class Config:
    def __init__(self, config, parent=None):
        self.parent_config = parent
        self.config = config
        for key in self.config:
            if type(self.config[key]) is dict:
                self.config[key] = Config(self.config[key], self)

    @staticmethod
    def load(path, config_names):
        # based on https://github.com/LAION-AI/Open-Assistant
        all_confs = {}
        no_conf = True

        for config_file in path.glob('*.yaml'):
            no_conf = False
            with config_file.open('r') as f:
                all_confs.update(recursive_lowercase_keys(yaml.safe_load(f)))

        if no_conf:
            print(f'ERROR: No yaml files found in {dir}')

        config = {}
        for name in config_names:
            if ',' in name:
                for n in name.split(','):
                    config.update(all_confs[n])
            else:
                config.update(all_confs[name])
        
        return Config(config, None)
    
    def __getattr__(self, name):
        if name in self.config:
            return self.config[name]
        elif self.parent_config is not None:
            return getattr(self.parent_config, name)
        else:
            raise AttributeError(f"Config has no attribute {name}.")
    
    def __getitem__(self, name):
        return getattr(self, name)
    
    def __contains__(self, name):
        return name in self.config

    def get(self, name, default=None):
        try:
            return self[name]
        except AttributeError:
            return default
