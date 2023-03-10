from torch.nn.utils.rnn import pad_sequence
import json
from pathlib import Path
from collections import OrderedDict


class MyCollate:
    '''
    class to add padding to the batches
    collat_fn in dataloader is used for post processing on a single batch. Like __getitem__ in dataset class is used on single example
    '''
    def __init__(self, pad_idx, batch_first):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
        
    
    #__call__: a default method
    ##   First the obj is created using MyCollate(pad_idx) in data loader
    ##   Then if obj(batch) is called -> __call__ runs by default
    def __call__(self, batch):
        #get all source indexed sentences of the batch
        source = [item[0] for item in batch] 
        #pad them using pad_sequence method from pytorch. 
        source = pad_sequence(source, batch_first=self.batch_first, padding_value = self.pad_idx) 
        
        #get all target indexed sentences of the batch
        target = [item[1] for item in batch] 
        #pad them using pad_sequence method from pytorch. 
        target = pad_sequence(target, batch_first=self.batch_first, padding_value = self.pad_idx)
        return source, target

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as fhandle:
        return json.load(fhandle, object_hook=OrderedDict)
    
def init_obj(config, name, module, *args, **kwargs):
    module_name = config[name]['type']
    module_args = dict(config[name]['args'])
    assert all(k not in module_args for k in kwargs)
    print(f"module name: {module_name}, module_args: {module_args}")
    
    return getattr(module, module_name)(*args, **module_args)