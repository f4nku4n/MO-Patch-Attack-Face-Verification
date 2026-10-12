import os
import torch
from PIL import Image
from torchvision import transforms

from factory import InceptionResnetV1, iresnet100

ALL_MODELS = ['vggface', 'webface', 'arcface', 'cosface']
THRESHOLD = 0.5  # same threshold as core/fitness.py

_to_tensor = transforms.ToTensor()


def get_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def img_size(model_name):
    """Same input sizes as main_ours.py."""
    return (160, 160) if model_name in ['vggface', 'webface'] else (112, 112)


def load_model(model_name, model_dir, device):
    """Same as factory.get_model, but works on any device (not only CUDA)."""
    if model_name == 'vggface':
        model, ckpt = InceptionResnetV1('vggface2'), f'{model_dir}/vggface2.pt'
    elif model_name == 'webface':
        model, ckpt = InceptionResnetV1('casia-webface'), f'{model_dir}/webface.pt'
    elif model_name == 'arcface':
        model, ckpt = iresnet100(), f'{model_dir}/arcface.pth'
    elif model_name == 'cosface':
        model, ckpt = iresnet100(), f'{model_dir}/cosface.pth'
    else:
        raise ValueError(model_name)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    return model.eval().to(device)


def img_path(img_dir, name, img_id):
    return os.path.join(img_dir, name, f'{name}_{str(img_id).zfill(4)}.jpg')


def load_img_tensor(img_dir, name, img_id, model_name):
    """Same preprocessing as main_ours.py: PIL RGB -> resize -> ToTensor (values in [0, 1])."""
    h, w = img_size(model_name)
    img = Image.open(img_path(img_dir, name, img_id)).convert('RGB').resize((h, w))
    return _to_tensor(img)


def list_identity_ids(img_dir, name):
    """All image ids available for one identity, e.g. Amanda_Beard_0003.jpg -> 3."""
    folder = os.path.join(img_dir, name)
    ids = []
    for f in os.listdir(folder):
        stem, ext = os.path.splitext(f)
        if ext.lower() != '.jpg' or not stem.startswith(name + '_'):
            continue
        tail = stem[len(name) + 1:]
        if tail.isdigit():
            ids.append(int(tail))
    return sorted(ids)


def read_pairs(path):
    """Read a pair file (first line is a header). Returns list of tab-split lines."""
    with open(path) as f:
        header = f.readline().strip()
        lines = [l.rstrip('\n').split('\t') for l in f if l.strip()]
    return header, lines


@torch.no_grad()
def embed(model, x, device):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return model(x.to(device))


@torch.no_grad()
def cos_sim(f1, f2):
    return torch.nn.functional.cosine_similarity(f1, f2, dim=1)