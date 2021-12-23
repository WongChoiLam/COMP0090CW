import h5py
from torch.utils.data import Dataset

class Oxpet_Dataset(Dataset):
    def __init__(self, img_path, binary_path, bboxes_path, masks_path):
        self.img_path = img_path
        self.binary_path = binary_path
        self.bboxes_path = bboxes_path
        self.masks_path = masks_path

    def __len__(self):
        with h5py.File(self.binary_path,"r") as f:
            key = list(f.keys())[0]
            return len(f[key])

    def __getitem__(self, idx):
        img = None
        binary = None
        bboxes = None
        masks = None

        with h5py.File(self.img_path,"r") as f:
            key = list(f.keys())[0]
            img = f[key][idx]
        with h5py.File(self.binary_path,"r") as f:
            key = list(f.keys())[0]
            binary = f[key][idx]
        with h5py.File(self.bboxes_path,"r") as f:
            key = list(f.keys())[0]
            bboxes = f[key][idx]
        with h5py.File(self.masks_path,"r") as f:
            key = list(f.keys())[0]
            masks = f[key][idx]
        
        return (img, binary, bboxes, masks)

# training_data = Oxpet_Dataset(os.path.join("datasets-oxpet", "train","images.h5"),os.path.join("datasets-oxpet", "train","binary.h5"),os.path.join("datasets-oxpet", "train","bboxes.h5"),os.path.join("datasets-oxpet", "train","masks.h5"))
# ox_dataloader = DataLoader(training_data, batch_size=32, shuffle= True,num_workers=4)