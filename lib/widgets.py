import os
import json
import pickle
import random
import pandas as pd
from IPython.display import Javascript


class Widgets:
    def __init__(self, id=None):
        self.id = id or format(random.randrange(16**8), '08x')
        self.handle = display(display_id=self.id)
        
    
class Hud(Widgets):
    def __init__(self, id=None):
        super().__init__(id=id)
        self.reset()

    def __setitem__(self, key, value):
        self.data[key] = value
        self.refresh()
    
    def __call__(self, data={}, index=None, reset=False):
        if reset:
            self.reset()
        self.update(data, index)
        self.refresh()
        
    def reset(self):
        self.index = ''
        self.data = {}
        
    def update(self, data={}, index=None):
        if index is not None:
            self.index = index
        self.data.update(data)
        
    def refresh(self):
        self.handle.update(pd.DataFrame({key: [self.data[key]] for key in self.data}, index=[self.index]))
        
        
class CopyButton(Widgets):
    def __init__(self, id=None):
        super().__init__(id=id)
        self.reset()
        
    def __setitem__(self, key, value):
        if key is None:
            self.data.append(value)
        else:
            self.data[key] = value
        self.refresh()
        
    def __call__(self, data=[], label=None, reset=False):
        if reset:
            self.reset()
        self.update(data, label)
        self.refresh()
    
    def reset(self):
        self.label = ''
        self.data = []
        
    def update(self, data=[], label=None):
        if label is not None:
            self.label = label
        self.data = data
        
    def refresh(self):
        sep = r'\t'
        self.handle.update(Javascript(f'''
            (async() => {{
                const button = document.createElement('button');
                try {{
                    button.textContent = "{self.label}";
                    document.body.appendChild(button);
                    await new Promise((resolve) => {{
                        button.addEventListener('click', () => {{
                            navigator.clipboard.writeText("{sep.join(self.data)}");
                            resolve();
                        }});    
                    }});
                }} finally {{ }}
            }})();
        '''))