import json


def save_json(json_data, filename):
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent="\t", separators=(',', ':'))


def load_json(json_path):
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    return json_data


def get_region_attributes(bnd):
    d = {}
    d['shape_attributes'] = {}

    d['shape_attributes']['index'] = bnd[0]
    d['shape_attributes']['x'] = bnd[1]
    d['shape_attributes']['y'] = bnd[2]
    d['shape_attributes']['width'] = bnd[3]
    d['shape_attributes']['height'] = bnd[4]

    return d


def to_json(img_path, d):
    filename = img_path

    js = {}
    js['filename']  = filename
    js['regions'] = []

    region_count = len(d)
    for i in range(0, region_count):
        ri = get_region_attributes(d[i])
        js['regions'].append(ri)

    return js


def compare_point(json_path, mx, my):
    d = load_json(json_path)
    selected = {}

    selected['filename'] = d['filename']
    selected['index'] = []
    
    for region in d['regions']:
        x = region['shape_attributes']['x']
        y = region['shape_attributes']['y']
        w = region['shape_attributes']['width']
        h = region['shape_attributes']['height']

        if x <= float(mx) <= x+w and y <= float(my) <= y+h:
            selected['index'].append(region['shape_attributes']['index'])

    print(selected['index'])
    if not selected['index']:
        print('List is empty')
        return None 
    else:
        for i in range(len(selected['index'])):
            print("{}'s contour is selected.".format(selected['index'][i]))
        return selected

'''
def resize_bndbox(json_path, mx, my):
    d = load_json(json_path)
    selected = {}

    selected['filename'] = d['filename']
    selected['index'] = []

    for region in d['regions']:
        x = region['shape_attributes']['x']
        y = region['shape_attributes']['y']
        w = region['shape_attributes']['width']
        h = region['shape_attributes']['height']

        if (mx = x and my = y) or (mx = x+w and my = y+h):
            pass
'''
