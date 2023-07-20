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
            list_file.append([filename, label, filename.split('_')[0]])
        return list_file
    else:
        print('mode is inappropriate.')
        return