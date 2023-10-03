# DOC Story Generation V2

This repository contains code for automatically generating stories of a few thousand words in length using LLMs, based on the same main ideas and overall structure as https://github.com/yangkevin2/doc-story-generation, but updated to work with newer open-source and chat-based LLMs. The code has been largely rewritten to make it easier to modify for experimentation.

## Installation

Tested using Python 3.9 but similar versions should also work.

```
pip install -r requirements.txt
pip install -e .
```

By default we use VLLM to serve models. 
You'll need to make a one-line change to the VLLM package to get their API server to work with logprobs requests that are used for reranking.
In your install of VLLM (you can find it using e.g., `pip show vllm`), find the line at https://github.com/vllm-project/vllm/blob/acbed3ef40f015fcf64460e629813922fab90380/vllm/entrypoints/openai/api_server.py#L177 (your exact line number might vary slightly depending on VLLM version) and change the `p` at the end to e.g., `max(p, -1e8)`. This will avoid an error related to passing jsons back from the server, due to json not handling inf values.

## Getting Started

We divide the story generation procedure into 3 steps, Premise, Plan, and Story. Everything will be run from the `scripts` directory:

```
cd scripts
```

Everything will read the information from the `defaults` configuration in `config.yaml` unless specified otherwise using the `--configs` flag.

See the corresponding `config.yaml` for details on options for each step of the pipeline. You'll have to fill in the particular model you want to use (marked TODO in each `config.yaml`). This system was mainly tested with LLaMA2-7B-Chat and the default options given; several other options are supported but not as heavily tested. You can also add new options directly to the config as needed; you can also see the main prompts in `prompts.json`.

By default we use VLLM to serve models. Start the server(s) for the models you're using (this will start them in the background). 

```
python start_servers.py --step {premise/plan/story}
```

Then run the generation pipeline.

```
python {premise/plan/story}/generate.py
```

By default, files are written to the `output/` folder. Premise and Plan are formatted as jsons which can be edited for human interaction.

After you're done with a given step, close your servers (this command also runs in the background). 

```
python close_servers.py
```

Note that `start_servers.py` relies on `close_servers.py` to delete the `server_configs.txt` file; just delete it manually before starting servers next time if you close servers in a different way. Alternatively, if memory allows, you can just keep all the servers alive simultaneously before closing them at the end, or reuse the servers between steps if you're using the same model by setting them to use the same `port` in `config.yaml` (it's fine if the sampling params differ).