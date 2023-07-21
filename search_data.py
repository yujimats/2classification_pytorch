import os

def search_data():
    path_input = os.path.join('dataset_toyota_cars')
    list_dir = [f for f in os.listdir(path_input) if os.path.isdir(os.path.join(path_input, f))]
    # print(len(list_dir))
    with open('./dataset_toyota_cars.csv', 'w') as logfile:
        logfile.write('dir_name,file_count\n')
    for dir in list_dir:
        path = os.path.join(path_input, dir)
        list_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        with open('./dataset_toyota_cars.csv', 'a') as logfile:
            logfile.write('{},{}\n'.format(dir, len(list_files)))

if __name__=='__main__':
    search_data()