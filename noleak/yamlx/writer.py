from .common import _get_default_heading
import yaml

BUFFER_SIZE = 1024

_cast = {
    list: repr,
    str: repr,
    int: repr,
    float: repr,
    type(None): lambda _: "null"
}

def _caster(entry):
    try:
        return map(lambda x: _cast[type(x)](x), entry)
    except KeyError as e:
        raise e

def _dump_list(entry, features, is_min, i):
    entry = _caster(entry)
    lines = []
    if features:
        is_min = is_min and i
        front =  '- - ' if is_min else  '- '
        for feature, val in zip(features, entry):
            if is_min:
                lines.append(front +  val)
                front = '  - ' #4 spaces
            else:
                lines.append(front + str(feature) + ': ' + val)
                front = '  '  #2 spaces
        return lines
    else:
        front =  '- - ' 
        for val in entry:
            lines.append(front +  val)
            front = '  - ' #4 spaces
        return lines

def _dump(entry, is_min, i):
    entry = _caster(entry)
    is_min = is_min and i
    lines = ""
    front =  '- - ' if is_min else  '- '
    for feature, val in zip(features, entry):
        if is_min:
            lines = lines + front + feature + ': ' + val + '\n'
            front = '  '  #2 spaces
        else:
            lines = lines + front +  val + '\n'
            front = '  - ' #4 spaces
    return lines

def write_meta(iof, meta, **kwargs):
    if meta:
        yaml.safe_dump(meta, iof, default_flow_style=False, **kwargs)
        #iof.write('\n---\n')
        #iof.write('\ndataset:\n')

def write_dataset_heading(iof):
    iof.write('\ndataset:\n')
    
def write(path, dataset=None, features=None, meta= None, is_min=True):
    assert all(map(lambda x: isinstance(x, str), features))
    is_add_features = (not is_min) and features
    is_min = is_min or not features
    with open(path, 'w') as f:
        if meta:
            yaml.safe_dump(meta, f, default_flow_style=False)
            dataset_heading = _get_default_heading()
            f.write('\n' + dataset_heading + ':\n')
    
    if dataset:
        batch = ""
        with open(path, 'a') as f:
            for i, entry in enumerate(dataset):
                assert len(entry) == len(features)
                lines = _dump(entry, is_min, i)
                batch = batch + lines
                if len(batch) > BUFFER_SIZE:
                    f.write(batch)
                    batch = ""
            f.write(batch)