# Copyright (c) Meta Platforms, Inc. and affiliates.

from storygen.common.util import *
from storygen.common.llm.llm import *
from storygen.premise.premise import Premise
from storygen.plan.setting import Setting
from storygen.plan.entity import *
from storygen.plan.outline import *


def generate_setting(plan, llm_client, setting_prompt, setting_config):
    plan.setting = Setting(llm_client.call_with_retry(
        setting_prompt.format(title=plan.premise.title, premise=plan.premise.premise),
        SamplingConfig.from_config(setting_config),
        filter=min_max_tokens_filter(0, setting_config['max_tokens']))[0]
    )
    logging.debug(f"Setting: {plan.setting.setting}")
    return plan


def generate_entities(plan, llm_client, entity_prompt, entity_config):
    def postprocess_name(names, **kwargs):
        return [name.strip(string.whitespace + string.punctuation) for name in names]
    def postprocess_entity_description(descriptions, **kwargs):
        responses = []
        for description in descriptions:
            if '\n' in description:
                responses.append((description.split('\n')[0].rstrip(), True))
            else:
                responses.append((description.rstrip(), False))
        return responses
    name_config, description_config = entity_config['name'], entity_config['description']
    name_prompt, description_prompt = entity_prompt['name'], entity_prompt['description']
    plan.entity_list = EntityList()
    has_next = True
    while has_next:
        entity_name = llm_client.call_with_retry(
            name_prompt.format(
                title=plan.premise.title, 
                premise=plan.premise.premise,
                setting=plan.setting.setting,
                previous_entities=plan.entity_list.print_with_full_names(),
                current_number=len(plan.entity_list) + 1
            ),
            SamplingConfig.from_config(name_config),
            postprocessor=postprocess_name,
            filter=word_filter([entity.name for entity in plan.entity_list] + ['full', 'Full', 'name', 'Name']) + \
                    min_max_tokens_filter(0, name_config['max_tokens'])
        )[0]
        entity_description, has_next = llm_client.call_with_retry(
            description_prompt.format(
                title=plan.premise.title, 
                premise=plan.premise.premise,
                setting=plan.setting.setting,
                previous_entities=str(plan.entity_list),
                current_number=len(plan.entity_list) + 1,
                name=entity_name
            ),
            SamplingConfig.from_config(description_config),
            postprocessor=postprocess_entity_description,
            filter=wrap_filter_for_tuple(list_next_number_format_filter() + \
                    min_max_tokens_filter(0, description_config['max_tokens']) + \
                    Filter(lambda s: s.endswith('.')))
        )[0]
        plan.entity_list.entities.append(Entity(entity_name, entity_description))
        if len(plan.entity_list) < entity_config['min_entities']:
            has_next = True
        elif len(plan.entity_list) >= entity_config['max_entities']:
            has_next = False
    return plan


def generate_outline(plan, llm_client, outline_prompt, outline_config):
    plan.outline = OutlineNode('', None)
    while True:
        try:
            node_to_expand = select_node_to_expand(plan.outline, outline_config)
        except StopIteration:
            break
        generate_node_subevents(node_to_expand, llm_client, outline_prompt, outline_config, plan)
        logging.debug(plan.outline)
    return plan


def generate_node_subevents(node, llm_client, outline_prompt, outline_config, plan):
    def event_postprocessor(events, has_next_indicator, current_number, **kwargs):
        responses = []
        for event in events:
            while '\n ' in event:
                event = event.replace('\n ', '\n') # avoid problems with spacing/tabs making us miss the has_next_indicator, e.g., \nii.
            while '\n\t' in event:
                event = event.replace('\n\t', '\n')
            has_next = has_next_indicator in event
            # logging.debug('has next: ' + has_next_indicator + ' ' + str(has_next) + ' ' + event)
            event = event.split(has_next_indicator)[0].rstrip()
            if event.lstrip().startswith('[') and ']' in event:
                event = event.split(']')[1].lstrip()
            event = event.lstrip().lstrip(':').lstrip()
            event = event.split('\n')[0].rstrip().split('(')[0].rstrip()
            event = event.split('Scene:')[0].rstrip()
            event = event.split('Characters:')[0].rstrip()
            
            event = event.lstrip()
            if len(event) > 0:
                if event[-1] not in ['.', '?', '!']:
                    if event[-1] in string.punctuation:
                        event = '' # cause this to be filtered out
                    else:
                        event += '.'
            if len(event.split()) > 0 and current_number in event.split()[0]:
                event = event[len(event.split()[0]):].lstrip()
            responses.append((event, has_next))
        return responses
    if node.depth() == 0:
        # slightly different handling for the first expansion at depth 0
        event_config = outline_config['event_depth_0']
        event_prompt = outline_prompt['event_depth_0']
    else:
        event_config = outline_config['event']
        event_prompt = outline_prompt['event']
    has_next = True
    while has_next:
        new_child = OutlineNode('', node)
        node.children.append(new_child)
        context_prefix, context_suffix = new_child.context(outline_config['context'])
        filter = wrap_filter_for_tuple(min_max_tokens_filter(0, event_config['max_tokens']) + word_filter(['[', 'TODO', ']', ':']))
        if len(node.children) >= outline_config['max_children']:
            filter = filter + wrap_filter_for_tuple(Filter(lambda t: not t[1])) # shouldn't continue past max children, so has_next should be false
        filter += wrap_filter_for_tuple(levenshtein_ratio_filter([n.text for n in node.root().depth_first_traverse(include_self=False)]))
        event, has_next = llm_client.call_with_retry(
            event_prompt.format(
                title=plan.premise.title,
                premise=plan.premise.premise,
                setting=plan.setting.setting,
                entities=str(plan.entity_list),
                formatted_current_number=new_child.number().rstrip(),
                stripped_current_number=new_child.number().strip(),
                context_prefix=context_prefix,
                context_suffix=context_suffix,
                predecessor_info=f'describing the beginning of "{new_child.predecessor().text}"' if len(node.children) == 1 else f'describing the conclusion of "{node.text}" after "{new_child.predecessor().text}"',
                successor_info=f'but before "{new_child.successor().text}"' if new_child.successor() is not None else 'The upcoming event(s) are the conclusion of the whole story, so make sure to wrap things up nicely.',
                preferred_max_children=outline_config['preferred_max_children'],
            ),
            SamplingConfig.from_config(event_config),
            postprocessor=partial(event_postprocessor, has_next_indicator='\n' + new_child.number(lookforward=1).strip(), current_number=new_child.number(lookforward=0).strip()),
            filter=filter,
        )[0]
        new_child.text = event
        generate_node_scene(
            new_child, 
            llm_client, 
            outline_prompt['scene'], 
            outline_config['scene'], 
            plan
        )
        generate_node_entities(
            new_child, 
            llm_client, 
            outline_prompt['entity_depth_0'] if node.depth() == 0 else outline_prompt['entity'],
            outline_config['entity_depth_0'] if node.depth() == 0 else outline_config['entity'],
            plan
        )
        logging.info(f"Newly generated node: {new_child}")
        if len(node.children) < outline_config['min_children']:
            has_next = True
        elif len(node.children) >= outline_config['max_children']:
            if has_next:
                logging.warning(f"Max children reached but model not done generating for this expansion")
            assert not has_next


def generate_node_scene(node, llm_client, scene_prompt, scene_config, plan):
    def scene_postprocessor(scenes, **kwargs):
        responses = []
        for scene in scenes:
            scene = scene.lstrip()
            scene = scene.split('\n')[0].rstrip()
            scene = scene.split('Characters:')[0].rstrip()
            scene = scene.split('Scene:')[-1].lstrip()
            if '"' in scene:
                scene = scene[:scene.index('"')]
            responses.append(scene)
        return responses
    context_prefix, context_suffix = node.context(scene_config['context'])
    node.scene = llm_client.call_with_retry(
        scene_prompt.format(
            title=plan.premise.title,
            premise=plan.premise.premise,
            setting=plan.setting.setting,
            entities=str(plan.entity_list),
            formatted_current_number=node.number().rstrip(),
            stripped_current_number=node.number().strip(),
            current_event=node.text,
            context_prefix=context_prefix,
            context_suffix=context_suffix,
        ),
        SamplingConfig.from_config(scene_config),
        postprocessor=scene_postprocessor,
        filter=min_max_tokens_filter(0, scene_config['max_tokens']) + word_filter(['[', 'TODO', ']', ':'])
    )[0]


def generate_node_entities(node, llm_client, entity_prompt, entity_config, plan):
    def entity_postprocessor(predicted_entities_lists, entity_list, already_detected_entities, **kwargs):
        responses = []
        for entities in predicted_entities_lists:
            entities = entities.split('\n')[0].rstrip()
            if entities.endswith('.'):
                entities = entities[:-1]
            entities = entities.split(', ')
            entities = [entity.strip() for entity in entities]
            entities = [entity for entity in entities if entity in [e.name for e in entity_list]]
            dedup_entities = []
            for entity in entities:
                if entity not in dedup_entities:
                    dedup_entities.append(entity)
            entities = [entity for entity in dedup_entities if entity not in already_detected_entities]
            responses.append(already_detected_entities + entities)
        # if all([len(response) == 0 for response in responses]):
        #     import pdb; pdb.set_trace()
        return responses
    context_prefix, context_suffix = node.context(entity_config['context'])
    logging.debug(node.text)
    logging.debug('detect entities:' + ', '.join(detect_entities(node.text, plan.entity_list)))
    detected_entities = detect_entities(node.text, plan.entity_list)
    try:
        node.entities = llm_client.call_with_retry(
            entity_prompt.format(
                title=plan.premise.title,
                premise=plan.premise.premise,
                setting=plan.setting.setting,
                entities=str(plan.entity_list),
                formatted_current_number=node.number().rstrip(),
                stripped_current_number=node.number().strip(),
                current_event=node.text,
                current_scene=node.scene,
                context_prefix=context_prefix,
                context_suffix=context_suffix,
                detected_entities=' ' + ', '.join(detected_entities) if len(detected_entities) > 0 else '',
            ),
            SamplingConfig.from_config(entity_config),
            postprocessor=partial(entity_postprocessor, entity_list=plan.entity_list, already_detected_entities=detected_entities),
            filter=Filter(lambda l: len(l) > 0),
            max_attempts=20 # TODO this call is disproportionately likely to fail
        )[0]
    except:
        # if this fails, just use the predecessor's entities
        logging.warning(f"Failed to generate entities for node {node.number()} with text: {node.text}; using predecessor's entities instead")
        node.entities = [e for e in node.predecessor().entities]
    logging.debug(f"Generated entities: {node.entities}")


def select_node_to_expand(outline, outline_config):
    if outline_config['expansion_policy'] == 'breadth-first':
        for node in outline.breadth_first_traverse(max_depth=outline_config['max_depth']-1):
            if len(node.children) == 0:
                return node
        raise StopIteration
    else:
        raise NotImplementedError