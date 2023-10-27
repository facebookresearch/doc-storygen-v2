# Copyright (c) Meta Platforms, Inc. and affiliates.

import json

class Premise:
    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            data = json.load(f)
            return Premise(data['title'], data['premise'])

    def __init__(self, title=None, premise=None):
        self.title = title
        self.premise = premise
    
    def __str__(self):
        return f'Title: {self.title}\n\nPremise: {self.premise}'
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'title': self.title,
                'premise': self.premise
            }, f, indent=4)