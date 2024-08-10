DATASET_HEADINGS = ['dataset', 'data']

def _is_dataset_heading(line):
    line_split= line.rstrip().split(':', 1)
    key = line_split[0].rstrip()
    if len(line_split) == 2 and not line_split[1]:
        return any([key == x for x in DATASET_HEADINGS])
    return False

def _get_default_heading():
    return DATASET_HEADINGS[0]