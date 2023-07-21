import os

def get_files_list(path_input, mode='dog&cats'):
    if mode == 'dog&cats':
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
    elif mode == 'toyota_cars':
        path_camry = os.path.join(path_input, 'camry')
        path_corolla = os.path.join(path_input, 'corolla')
        dict_list_filenames = {
            'camry': os.listdir(path_camry),
            'corolla': os.listdir(path_corolla)
        }
        list_file = []
        for key, val in dict_list_filenames.items():
            if key == 'camry':
                label = 0 # camry
                for filename in val:
                    list_file.append([os.path.join(path_camry, filename), label, key])
            elif key == 'corolla':
                label = 1 # corrola
                for filename in val:
                    list_file.append([os.path.join(path_corolla, filename), label, key])
        return list_file
    else:
        print('mode is inappropriate.')
        return