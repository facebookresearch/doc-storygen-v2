# Copyright (c) Meta Platforms, Inc. and affiliates.

import string

try:
    from nltk.corpus import stopwords
    _ = stopwords.words('english') # check that it's loaded
except:
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords


class Entity:
    def __init__(self, name, description):
        self.name = name
        self.description = description


class EntityList:
    def __init__(self, entities=None):
        self.entities = entities if entities is not None else []
    
    def __len__(self):
        return len(self.entities)
    
    def __str__(self):
        # return numbered list of entity names with descriptions
        return '\n\n'.join([f'{i+1}. {entity.name}: {entity.description}' for i, entity in enumerate(self.entities)])
    
    def print_with_full_names(self):
        # return numbered list of entity names with descriptions
        return '\n\n'.join([f'{i+1}. Full Name: {entity.name}\n\nDescription: {entity.description}' for i, entity in enumerate(self.entities)])

    def __iter__(self):
        return iter(self.entities)

    def __getitem__(self, index):
        return self.entities[index]
    
    def get_entity_by_name(self, name):
        for entity in self.entities:
            if entity.name == name:
                return entity
        raise ValueError(f'EntityList has no entity named {name}.')


def detect_entities(event, entity_list):
    detected_entities = []
    for entity in entity_list:
        for name in entity.name.split():
            name = name.lower()
            # check name not in any other entity names, for disambiguation
            if name not in ''.join([other_entity.name.lower() for other_entity in entity_list if other_entity != entity]):
                if name in event.lower() and name not in stopwords.words('english'):
                    detected_entities.append(entity.name)
                    break
    return detected_entities