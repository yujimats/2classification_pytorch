import os

def get_files_list_pets(path_input, label_0=None, label_1=None):
    list_filenames = os.listdir(path_input)
    list_file = []
    for filename in list_filenames:
        if ('newfoundland' in filename) or ('pomeranian' in filename):
            label = 0 # dog
        elif ('Abyssinian' in filename) or ('Bombay' in filename):
            label = 1 # cat
        else:
            continue
        list_file.append([os.path.join(path_input, filename), label, filename.split('_')[0]])
    return list_file

def get_files_list_toyota_cars(path_input, label_0=None, label_1=None):
    elif mode == 'toyota_cars':
    if (label_1 == None) or (label_1==None):
        print('if mode toyota_cars, select label name as label_0 and label_1')
        return
    path_label_0 = os.path.join(path_input, label_0)
    path_label_1 = os.path.join(path_input, label_1)
    dict_list_filenames = {
        label_0: os.listdir(path_label_0),
        label_1: os.listdir(path_label_1)
    }
    list_file = []
    for key, val in dict_list_filenames.items():
        if key == label_0:
            label = 0
            for filename in val:
                list_file.append([os.path.join(path_label_0, filename), label, key])
        elif key == label_1:
            label = 1
            for filename in val:
                list_file.append([os.path.join(path_label_1, filename), label, key])
    return list_file