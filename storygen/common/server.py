# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import os

LOCALHOST = 'http://localhost'
DEFAULT_PORT = 8000


class ServerConfig:
    def __init__(self, engine, host, port, server_type):
        self.engine = engine
        self.host = host
        self.port = port
        self.server_type = server_type
    
    @staticmethod
    def from_config(config):
        return ServerConfig(
            engine=config['engine'],
            host=config['host'],
            port=config.get('port', DEFAULT_PORT),
            server_type=config['server_type']
        )
    
    @staticmethod
    def from_json(json_str):
        return ServerConfig(**json.loads(json_str))

    def json(self):
        return json.dumps({
            'engine': self.engine,
            'host': self.host,
            'port': self.port,
            'server_type': self.server_type
        })

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __hash__(self):
        return hash((self.engine, self.host, self.port, self.server_type))

    def __eq__(self, other):
        return (self.engine, self.host, self.port, self.server_type) == (other.engine, other.host, other.port, other.server_type)


def start_server(config):
    # log port to file with appending
    if os.path.exists('server_configs.txt'):
        with open('server_configs.txt', 'r') as f:
            existing_configs = f.read().split('\n')
            existing_configs = [ServerConfig.from_json(config_str) for config_str in existing_configs if config_str != '']
    else:
        existing_configs = []
    if config not in existing_configs:
        with open('server_configs.txt', 'a') as f:
            f.write(config.json() + '\n')
    else:
        logging.info(f"Server for {config['engine']} already started.")
        return
    if config['host'] == LOCALHOST and config['server_type'] == 'vllm':
        logging.info(f"Starting vllm server for {config['engine']} on port {config['port']}... (it's ready when it says \"Uvicorn running\")")
        # run vllm openai-interface server
        # try:
        os.system(f"python -u -m vllm.entrypoints.openai.api_server \
                        --model {config['engine']} \
                        --port {config['port']} &")
        #     logging.info(f"Started vllm server for {config['engine']} on port {config['port']}!")
        # except:
        #     logging.warning(f"Failed to start vllm server for {config['engine']} on port {config['port']}. Is the server already running?")
    else:
        logging.info(f"Not starting server for {config['engine']} (not localhost or not vllm).")
