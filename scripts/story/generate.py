# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import pickle

from pathlib import Path

from storygen.common.llm.llm import LLMClient
from storygen.common.llm.prompt import load_prompts
from storygen.plan.plan import Plan
from storygen.story.story_writer import *
from storygen.common.config import Config
from storygen.common.util import *

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', default=['defaults'])
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = Config.load(Path(dir_path), args.configs)
    init_logging(config['logging_level'])

    plan = Plan.load(config['plan_path'])
    prompts = load_prompts(Path(dir_path))

    client = LLMClient()
    
    story = generate_story(
        plan, 
        config['model']['story'], 
        prompts['story'], 
        client, 
        intermediate_save_prefix=config['intermediate_prefix'],
        delete_old_intermediates=config['delete_old_intermediates'],
    )[0]

    logging.info(f'Generated story: {story}')

    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
    story.save(config['output_path'])

    os.makedirs(os.path.dirname(config['output_pkl']), exist_ok=True)
    with open(config['output_pkl'], 'wb') as f:
        pickle.dump(story, f)