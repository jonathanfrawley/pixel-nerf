import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

import util
import torch
import numpy as np
from model import make_model
from render import NeRFRenderer
import torchvision.transforms as T
import tqdm
import imageio

import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize

from scipy.ndimage import rotate

from matplotlib import pyplot as plt
from pydicom import dcmread

from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt

# hyperparams
elevation = 0.0
radius = 1.3
z_near, z_far = 0.8, 1.8
num_views = 24
focal = 131.25
W = H = 512

gif = True
device = 'cuda'
focal = torch.tensor(focal, dtype=torch.float32, device=device)
output = os.path.join(ROOT_DIR, "output")

def normalize(arr):
    denominator = arr.max() - arr.min()
    if denominator == 0:
        return arr
    return (arr - arr.min()) / denominator

arrs = []
for i in range(216):
    path = f"../data/manifest-1612365584013/MIDRC-RICORD-1B/MIDRC-RICORD-1B-419639-000340/01-18-2005-CT CHEST HIGH RESOLUTION-06379/2.000000-SUPINE CHEST RECON 12-09859/1-{i+1:03}.dcm"
    ds = dcmread(path)
    arr = ds.pixel_array
    arrs.append(arr)

arr = np.array(arrs) # 216, 512, 512
arr = np.swapaxes(arr,0,1)
print(arr.shape)
x_lim, y_lim, z_lim = 511, 215, 511

_coord_to_blender = util.coord_to_blender()
_coord_from_blender = util.coord_from_blender()

print("Generating rays")
render_poses = torch.stack(
    [
        _coord_from_blender @ util.pose_spherical(angle, elevation, radius)
        for angle in np.linspace(-180, 180, num_views + 1)[:-1]
    ],
    0,
)

class CTImage(torch.nn.Module):
    def __init__(self, img):
        super().__init__()
        # For now boudning the HU values to make bone prominent then normalizing
        # Should instead use Beer's law: I=I_0 exp(-int_0^D \mu(x)dx) where \mu is the linear attenuation coefficient of the material related to the HU number
        img = img.clamp(800, 2000)
        img = normalize(img)
        self.img = img
    
    def forward(self, xyz, coarse=True, viewdirs=None, far=False):
        # xyz appear to be roughly between -1.7 and 1.7 (although it doesn't appear to exactly be in that range, not sure what it's meant to be in)
        # Scaling so projected images show the whole body.
        xyz = xyz.squeeze(0)
        xyz = (xyz + 1) / 2

        xyz[:,0] *= x_lim
        xyz[:,1] *= y_lim
        xyz[:,2] *= z_lim
        xyz = xyz.long().transpose(0,1)#.numpy()

        # get rows where values are out of bounds and put them back in bounds
        mask = (xyz[0,:]<0) | (xyz[1,:]<0) | (xyz[2,:]<0) | (xyz[0,:]>x_lim) | (xyz[1,:]>y_lim) | (xyz[2,:]>z_lim)
        xyz[:,mask] = 0

        sigma = self.img[tuple(xyz)]
        # Anything out of bounds set back to 0
        sigma[mask] = 0.0
        sigma = sigma.reshape(1, -1, 1)
        rgb = torch.ones(sigma.size(0), sigma.size(1), 3).to(device)
        return torch.cat((rgb, sigma), dim=-1).to(device)


image = CTImage(torch.tensor(arr).to(device))
renderer = NeRFRenderer(
    n_coarse=64, n_fine=32, n_fine_depth=16, depth_std=0.01, sched=[], white_bkgd=False, eval_batch_size=50000
).to(device=device)
render_par = renderer.bind_parallel(image, [0], simple_output=True).eval()

render_rays = util.gen_rays(render_poses, W, H, focal, z_near, z_far).to(device=device)

all_rgb_fine = []
for rays in tqdm(torch.split(render_rays.view(-1, 8), 80000, dim=0)):
    rgb, _depth = render_par(rays[None])
    all_rgb_fine.append(rgb[0])
_depth = None
rgb_fine = torch.cat(all_rgb_fine)
frames = (rgb_fine.view(num_views, H, W, 3).cpu().numpy() * 255).astype(
    np.uint8
)

im_name = "raw_data"

frames_dir_name = os.path.join(output, im_name + "_frames")
os.makedirs(frames_dir_name, exist_ok=True)

for i in range(num_views):
    frm_path = os.path.join(frames_dir_name, "{:04}.png".format(i))
    imageio.imwrite(frm_path, frames[i])

if gif:
    vid_path = os.path.join(output, im_name + "_vid.gif")
    imageio.mimwrite(vid_path, frames, fps=24)
else:
    vid_path = os.path.join(output, im_name + "_vid.mp4")
    imageio.mimwrite(vid_path, frames, fps=24, quality=8)
print("Wrote to", vid_path)