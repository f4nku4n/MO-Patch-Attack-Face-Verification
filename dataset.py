import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class LFW(Dataset):        
    def __init__(self, IMG_DIR, MASK_DIR, PAIR_PATH,
                 transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((160, 160))])):
        with open(PAIR_PATH, "r") as f:
            f.readline()
            lines = [line.strip().split("\t") for line in f.readlines()]
         
        self.lines = lines
        self.IMG_DIR = IMG_DIR
        self.MASK_DIR = MASK_DIR
        self.transform = transform
         
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        if len(line) == 3:
            first_identity_name, first_id, second_id = line
            second_identity_name = first_identity_name
            label = 0           
        elif len(line) == 4:
            first_identity_name, first_id, second_identity_name, second_id = line
            label = 1
        else:
            raise ValueError
        
        first_name = f"{first_identity_name}_{first_id.zfill(4)}.jpg"
        first_path = os.path.join(self.IMG_DIR, first_identity_name, first_name)

        second_name = f"{second_identity_name}_{second_id.zfill(4)}.jpg"
        second_path = os.path.join(self.IMG_DIR, second_identity_name, second_name)

        first_image = Image.open(first_path).convert("RGB")
        second_image = Image.open(second_path).convert("RGB")

        if self.transform:
            first_image = self.transform(first_image)
            second_image = self.transform(second_image) 

        return first_image, second_image, label
