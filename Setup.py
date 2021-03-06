import h5py
import os

def rewrite_h5py(dataset_path):
    os.makedirs(f'{dataset_path}-rewritten',exist_ok=True)
    for directory in os.listdir(f'{dataset_path}'):
        if directory.endswith('md'):
            continue
        os.makedirs(os.path.join(f'{dataset_path}-rewritten',directory),exist_ok=True)
        for file in os.listdir(os.path.join(dataset_path,directory)):
            all_data = None
            with h5py.File(os.path.join(dataset_path,directory,file),"r") as f:
                key = list(f)[0]
                all_data = f[key][()]
            with h5py.File(os.path.join(f'{dataset_path}-rewritten',directory,file), "w") as f:
                for i,data in enumerate(all_data):
                    f.create_dataset(f'{i}', data=data, compression='gzip')

if __name__ == '__main__':
    import time
    print('Rewrite Started...')
    start = time.time()
    rewrite_h5py('datasets-oxpet')
    end = time.time()
    print(f'Rewrite has been done in {end-start:.2f} seconds.')