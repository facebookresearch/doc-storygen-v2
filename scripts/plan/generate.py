# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os

from pathlib import Path

from storygen.common.llm.llm import LLMClient
from storygen.common.llm.prompt import load_prompts
from storygen.premise.premise import Premise
from storygen.plan.plan import Plan
from storygen.plan.plan_writer import *
from storygen.common.config import Config
from storygen.common.util import *

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', default=['defaults'])
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = Config.load(Path(dir_path), args.configs)
    init_logging(config['logging_level'])

    premise = Premise.load(config['premise_path'])
    prompts = load_prompts(Path(dir_path))

    client = LLMClient()

    plan = Plan(premise)

    generate_setting(plan, client, prompts['setting'], config['model']['setting'])
    logging.info(f'Generated setting: {plan.setting}')

    success = False
    for i in range(config['model']['entity']['max_attempts']):
        try:
            generate_entities(plan, client, prompts['entity'], config['model']['entity'])
            success = True
            break
        except:
            logging.warning(f'Failed to generate entities, retrying ({i+1}/{config["model"]["entity"]["max_attempts"]})')
    if not success:
        raise Exception('Failed to generate entities')
    logging.info(f'Generated entities: {plan.entity_list}')

    success = False
    for i in range(config['model']['outline']['max_attempts']):
        # TODO retry mechanism could be more sophisticated if needed, e.g. beam search or MCTS, similar to how we do it in generate_story
        try:
            generate_outline(plan, client, prompts['outline'], config['model']['outline'])
            success = True
            break
        except:
            logging.warning(f'Failed to generate outline, retrying ({i+1}/{config["model"]["outline"]["max_attempts"]})')
    if not success:
        raise Exception('Failed to generate outline')
    
    logging.info(f'Generated plan: {plan}')

    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
    plan.save(config['output_path'])