# DOC Story Generation V2

This repository contains code for automatically generating stories of a few thousand words in length using LLMs, based on the same main ideas and overall structure as https://github.com/yangkevin2/doc-story-generation, but substantially modified to work with newer open-source and chat-based LLMs. The main goal of this rewrite was simplicity, to make it easier to modify / build upon the codebase.

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

See the corresponding `config.yaml` for details on options for each step of the pipeline. You'll have to fill in the particular model you want to use (marked TODO in each `config.yaml`). This system was mainly tested with LLaMA2-7B-Chat and ChatGPT, with the default options given; several other options are supported but not as heavily tested. When changing the model, make sure you also change `server_type` and `prompt_format` as needed. You can also add new options directly to the config as needed; you can also see the main prompts in `prompts.json`.

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

## Human Evaluation

We performed human evaluation on 7000 generated story plot pairs, which can be found [here](https://dl.fbaipublicfiles.com/doc-storygen/story_annotation_v2.json). We take [oasst-3b](https://arxiv.org/abs/2304.07327) as the basic LLM and generate two different plots for the same premise. Each annotator was asked to compare two plots (Plot A and Plot B)  and to answer the following questions:

```
    Q1: Which story plot is more interesting to you overall?
    Q2: Please explain your answer to Q1.
    Q3: In your opinion, which one of the plots above could generate a more interesting book or movie (when a full story is written based on it)?
    Q4: Which story plot created more suspense and surprise?
    Q5: Which story's characters or events do you identify with or care for more?
    Q6: Which story has a better ending?
```


For Q2, they were required to provide a short text description for Q1. For other questions, they chose from four given options: *Plot A, Plot B, Both are good, Both are bad*. The description of the latter two options differ in different questions. We list some statistics below:

* 6931 uniq Premise, 60.6 average words 
* 403 uniq Annotator, 8 median annotations    
* 7000 uniq plan pair, 619.5 average words 
* 6993 uniq response to Q2, 69.0 average words 

And here are the results of our human evaluation.
|                                                                                    | Plot A | Plot B | Both are good | Both are bad |
| -----------------------------------------------------------------------------------| ------ | ------ | ------------- | ------------ |
| Q1. Which story plot is more interesting to you overall?                           | 31.90% | 41.54% | 11.84%        | 14.71%       |
| Q3. In your opinion, which one of the plots above could generate a more interesting book or movie (when a full story is written based on it)? | 30.81% | 40.69% | 14.37%        | 14.13%       |
| Q4. Which story plot created more suspense and surprise?                           | 28.57% | 38.34% | 13.04%        | 20.04%       |
| Q5. Which story’s characters or events do you identify with or care for more?      | 30.31% | 38.60% | 14.09%        | 17.00%       |
| Q6. Which story has a better ending?                                               | 30.20% | 36.76% | 9.04%         | 24.00%       |

## Some Known Issues / Potential Improvements

- When start multiple model servers for different models, we should allocate them to different GPUs or load on multi-GPU as needed. 
- During plan generation, the model likes to overgenerate characters when determining which characters appear in a given plot point.
- Diversity of premise / plan ideas is kind of bad when using chat models, since they like to generate the same ideas over and over. Can try to increase temperature, or other ways to increase diversity, ideally without sacrificing quality.
- We should implement vaguest-first expansion (using a model to predict which node is the most vague) rather than breadth-first expansion when creating the outline during plan generation.
- We should test `text-davinci-003` for generating passages during the story generation stage since it's a completion model; the OpenAI Chat API is very limiting.
- Some model types and some more obscure options in the code aren't well-tested. Please let us know if you run into any issues.


## License

This repo is licensed under the Apache 2.0 License. See the LICENSE file for details.