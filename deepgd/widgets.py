import random
import pandas as pd
from abc import ABC, abstractmethod
from IPython.display import Javascript


class Widget:
    def __init__(self, id=None, data=None, title=None):
        self.id = id or format(random.randrange(16**8), '08x')
        self.handle = display(display_id=self.id)
        self.reset()
        self(data=data, title=title)
        
    def __setitem__(self, key, value):
        self.data[key] = value
        self.refresh()
        
    def __call__(self, data=None, title=None):
        if data is not None:
            self.data = data
        if title is not None:
            self.title = title
        self.refresh()
        return self
        
    def reset(self):
        self.data = {}
        self.title = ""
        self.refresh()
        return self
        
    def refresh(self):
        self.handle.update(repr(self))
        return self
        
    
class Hud(Widget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def append(self, data={}, title=None):
        self.data.update(data)
        self(title=title)
        return self
        
    def __repr__(self):
        return pd.DataFrame({key: [self.data[key]] for key in self.data}, index=[self.title])
        
        
class CopyButton(Widget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def append(self, data=[], title=None):
        self.data += data
        self(title=title)
        return self
        
    def __repr__(self):
        sep = r'\t'
        return Javascript(f'''
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
        ''')