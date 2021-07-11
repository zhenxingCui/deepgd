import os
import json
import pickle
import random
import pandas as pd
from pprint import pformat

def cache(cache_arg_key='cache', 
          cache_prefix_arg_key='cache_prefix', 
          cache_suffix_arg_key='cache_suffix'):
    def decorator(fn):
        def get_repr(obj):
            if type(obj) in {type(None), int, float, bool, str}:
                return str(obj)
            if callable(obj):
                return obj.__name__
            else:
                return type(obj).__name__
        def get_signature(fn, args, kwargs):
            vals = tuple(map(get_repr, args))
            kvs = tuple(map(lambda kv: f'{kv[0]}={get_repr(kv[1])}', kwargs.items()))
            arg_str = ','.join(map(str, vals + kvs))
            return f'{fn.__name__}({arg_str})'
        def cache_hit(file, args, kwargs):
            print(f"Load from '{file}'")
            return pickle.load(open(file, 'rb'))
        def cache_miss(file, args, kwargs):
            result = fn(*args, **kwargs)
            print(f"Save to '{file}'")
            pickle.dump(result, open(file, 'wb'))
            return result
        def wrapped(*args, **kwargs):
            cache_file = None
            cache_arg_val = kwargs.pop(cache_arg_key, False)
            cache_prefix_arg_val = kwargs.pop(cache_prefix_arg_key, './')
            cache_suffix_arg_val = kwargs.pop(cache_suffix_arg_key, '.pickle')
            if cache_arg_val:
                if type(cache_arg_val) is bool:
                    signature = get_signature(fn, args, kwargs)
                    cache_file = f'{cache_prefix_arg_val}{signature}{cache_suffix_arg_val}'
                elif type(cache_arg_val) is str:
                    cache_file = f'{cache_prefix_arg_val}{cache_arg_val}{cache_suffix_arg_val}'
                else:
                    raise Exception
            return (fn(*args, **kwargs) if not cache_arg_val
                    else cache_hit(cache_file, args, kwargs) if os.path.isfile(cache_file) 
                    else cache_miss(cache_file, args, kwargs))
        return wrapped
    return decorator


# TODO: setitem setattr
class Config:
    class Store:
        def __init__(self, data: dict):
            def wrap(kvpair):
                key, value = kvpair
                if type(value) is dict:
                    return key, Config.Store(data=value)
                return kvpair
            self.__dict__ = dict(map(wrap, data.items()))

        def __getitem__(self, item):
            def unwrap(kvpair):
                key, value = kvpair
                if type(value) is Config.Store:
                    return key, value[...]
                return kvpair
            if item is ...:
                return dict(map(unwrap, self.__dict__.items()))
            return self.__dict__[item]

        def __repr__(self):
            return pformat(self.__dict__)#, sort_dicts=False)

        def __str__(self):
            return str(self.__dict__)
        
    def __init__(self, file):
        self.file = file
        
    def __getitem__(self, item):
        data = Config.Store(json.load(open(self.file)))
        if item is None:
            return data
        return data[item]
    
    def __getattr__(self, attr):
        return self[attr]
    
    def __repr__(self):
        return pformat(self[None])#, sort_dicts=False)
    
    def __str__(self):
        return str(self[None])
    

class StaticConfig:
    class Store:
        def __init__(self, data: dict):
            def wrap(kvpair):
                key, value = kvpair
                if type(value) is dict:
                    return key, StaticConfig.Store(data=value)
                return kvpair
            self.__dict__ = dict(map(wrap, data.items()))

        def __getitem__(self, item):
            def unwrap(kvpair):
                key, value = kvpair
                if type(value) is StaticConfig.Store:
                    return key, value[...]
                return kvpair
            if item is ...:
                return dict(map(unwrap, self.__dict__.items()))
            return self.__dict__[item]

        def __repr__(self):
            return pformat(self.__dict__)#, sort_dicts=False)

        def __str__(self):
            return str(self.__dict__)
        
    def __init__(self, data):
        self.data = StaticConfig.Store(data)
        
    def __getitem__(self, item):
        if item is None:
            return self.data
        return self.data[item]
    
    def __getattr__(self, attr):
        return self[attr]
    
    def __repr__(self):
        return pformat(self[None])#, sort_dicts=False)
    
    def __str__(self):
        return str(self[None])
    
    
class Hud:
    def __init__(self, id=None):
        self.id = id or format(random.randrange(16**8), '08x')
        self.handle = display(display_id=self.id)

    def __call__(self, data):
        self.handle.update(pd.DataFrame(data))