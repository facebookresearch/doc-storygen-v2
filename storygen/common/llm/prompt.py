# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging

from langchain.prompts import PromptTemplate


warned_prompt_format = {'openai_response_prefix': False}


def format_langchain_prompt(langchain_prompt, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k in langchain_prompt.input_variables}
    return langchain_prompt.format(**kwargs)


class TemplatePromptBuilder:
    def __init__(self, base_dict):
        self.instruction = PromptTemplate.from_template(template=base_dict['instruction'],)
        self.system_message = PromptTemplate.from_template(template=base_dict['system_message'],) if 'system_message' in base_dict else None
        self.response_prefix = PromptTemplate.from_template(template=base_dict['response_prefix'],) if 'response_prefix' in base_dict else None
        self.output_prefix = PromptTemplate.from_template(template=base_dict['output_prefix'],) if 'output_prefix' in base_dict else None

    def format(self, **kwargs):
        return PromptBuilder(self, **kwargs)


class PromptBuilder:
    def __init__(self, template_prompt_builder, **kwargs):
        self.instruction = format_langchain_prompt(template_prompt_builder.instruction, **kwargs)
        self.system_message = format_langchain_prompt(template_prompt_builder.system_message, **kwargs) \
            if template_prompt_builder.system_message is not None else None
        self.response_prefix = format_langchain_prompt(template_prompt_builder.response_prefix, **kwargs) \
            if template_prompt_builder.response_prefix is not None else None
        self.output_prefix = format_langchain_prompt(template_prompt_builder.output_prefix, **kwargs) \
            if template_prompt_builder.output_prefix is not None else None

        # self.warned = {'openai_response_prefix': False}

    def render_for_llm_format(self, prompt_format):
        if prompt_format not in ['openai-chat', 'llama2-chat', 'none']:
            raise NotImplementedError(f"Prompt format {prompt_format} not implemented.")

        prompt = self.instruction.format().lstrip()

        if prompt_format == 'openai-chat':
            if self.response_prefix is not None:
                global warned_prompt_format
                if warned_prompt_format['openai_response_prefix']:
                    logging.warning(f"Response prefix is not supported for prompt format {prompt_format}. Appending to end of instruction instead.")
                    warned_prompt_format['openai_response_prefix'] = True
                prompt += '\n\n\n\nThe output is already partially generated. Continue from:\n\n' + self.response_prefix.format()
            messages = [{'role': 'user', 'content': prompt}]
            if self.system_message is not None:
                messages = [{'role': 'system', 'content': self.system_message.format()}] + messages
            return messages
        
        else:
            if prompt_format == 'llama2-chat':
                prompt = '[INST]'
                if self.system_message is not None:
                    prompt += ' <<SYS>>\n' + self.system_message.format() + '\n<</SYS>>\n\n'
                else:
                    prompt += ' '
                prompt += self.instruction.format()
                prompt += '[/INST]' if self.instruction.format()[-1] == ' ' else ' [/INST]'
                if self.response_prefix is not None:
                    prompt += self.response_prefix.format() if self.response_prefix.format()[0] == ' ' else ' ' + self.response_prefix.format()
            else:
                if self.system_message is not None:
                    prompt = self.system_message.format() + '\n\n\n\n' + prompt
                if self.response_prefix is not None:
                    prompt = prompt + '\n\n\n\n' + self.response_prefix.format()
            return prompt


def load_prompts(path):
    with open(path / 'prompts.json') as f:
        prompts = json.load(f)
    _create_prompt_templates(prompts)
    return prompts


def _create_prompt_templates(prompts):
    # recursively traverse prompts until you find a dict containing "instruction" key, and make TemplatePromptBuilder objects
    for key in prompts:
        assert isinstance(prompts[key], dict)
        if 'instruction' not in prompts[key]:
            _create_prompt_templates(prompts[key])
        else:
            prompts[key] = TemplatePromptBuilder(prompts[key])