# Copyright (c) Meta Platforms, Inc. and affiliates.

from storygen.common.llm.llm import SamplingConfig
from storygen.common.util import min_max_tokens_filter


def generate_title(premise_object, title_prompts, title_config, llm_client):
    title = llm_client.call_with_retry(
        title_prompts.format(), 
        SamplingConfig.from_config(title_config),
        filter=min_max_tokens_filter(0, title_config['max_tokens'])
    )[0]
    premise_object.title = title
    return premise_object


def generate_premise(premise_object, premise_prompts, premise_config, llm_client):
    premise = llm_client.call_with_retry(
        premise_prompts.format(title=premise_object.title), 
        SamplingConfig.from_config(premise_config),
        filter=min_max_tokens_filter(0, premise_config['max_tokens'])
    )[0]
    premise_object.premise = premise
    return premise_object