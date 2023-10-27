# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections.abc import Sequence

class Passage:
    def __init__(self, text, aux_info=None):
        self.text = text
        self.aux_info = aux_info if aux_info is not None else {}
    
    def __str__(self):
        return self.text


class OutlineNodePassageList:
    def __init__(self, outline_node, passages=None):
        self.outline_node = outline_node
        self.passages = passages if passages is not None else []
    
    def __len__(self):
        return len(self.passages)

    def __str__(self):
        return ''.join([str(passage) for passage in self.passages])
    
    def append(self, passage):
        self.passages.append(passage)
    
    def aux_attr_list(self, attr):
        return [passage.aux_info[attr] for passage in self.passages]


class Story:
    def __init__(self, plan, passage_lists=None):
        self.plan = plan
        self.passage_lists = passage_lists if passage_lists is not None else []
    
    def __len__(self):
        return len(self.passage_lists)
        # return sum([len(passage_list) for passage_list in self.passage_lists])
    
    def __str__(self):
        return ''.join([str(passage_list) for passage_list in self.passage_lists])
    
    def save(self, path):
        with open(path, 'w') as f:
            f.write(str(self))
    
    def copy_append_list(self, passage_list):
        return Story(self.plan, self.passage_lists + [passage_list])
    
    def copy_append_passage(self, passage):
        return Story(self.plan, self.passage_lists[:-1] + [OutlineNodePassageList(self.passage_lists[-1].outline_node, self.passage_lists[-1].passages + [passage])])
    
    def rendered_nodes(self):
        return [passage_list.outline_node for passage_list in self.passage_lists]
    
    def final_passage_aux_attr(self, attr):
        return self.passage_lists[-1].passages[-1].aux_info[attr]

    def passages(self):
        return [passage for passage_list in self.passage_lists for passage in passage_list.passages]
    
    def right_truncate(self, stop, allow_delete_passage_lists=False):
        # truncate passages from the right until we see the stop sequence
        for i in range(len(self.passage_lists) - 1, -1, -1):
            for j in range(len(self.passage_lists[i].passages) - 1, -1, -1):
                if stop in self.passage_lists[i].passages[j].text:
                    self.passage_lists[i].passages[j].text = stop.join(self.passage_lists[i].passages[j].text.split(stop)[:-1])
                    return self
                else:
                    # delete this passage
                    self.passage_lists[i].passages = self.passage_lists[i].passages[:j]
            if allow_delete_passage_lists:
                # delete this passage list
                self.passage_lists = self.passage_lists[:i]
            else:
                # stop here even though we didn't find the stop sequence
                return self


class StoryBeam(Sequence):
    def __init__(self, stories=None):
        self.stories = stories if stories is not None else []
        if len(self.stories) > 0:
            # make sure all stories same length
            assert len(set([len(story) for story in self.stories])) == 1
    
    def __len__(self):
        return len(self.stories)
    
    def __getitem__(self, index):
        return self.stories[index]
    
    def rendered_nodes(self):
        return self.stories[0].rendered_nodes() if len(self.stories) > 0 else set()
    
    def right_truncate(self, stop, allow_delete_passage_lists=False):
        for story in self.stories:
            story.right_truncate(stop, allow_delete_passage_lists=allow_delete_passage_lists)
        return self