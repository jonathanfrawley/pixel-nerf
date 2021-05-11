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

## Hyperparams
elevation = 0.0
num_views = 48
## Radius and focal length set as in 2.4.f as here https://iopscience.iop.org/article/10.1088/0031-9155/45/10/305/pdf
radius = 100 # how far away the x-ray source is from centre of the the patient in cm
focal = 140 # how far away the x-ray source is from the detector in cm
## Resolution and sensor size can be set independently
W = H = width_pixels = height_pixels = 512 # number of pixels over width/height
width = height = 60 # width/height of detector in cm

gif = True
device = 'cuda'
output = os.path.join(ROOT_DIR, "output")

def normalize(arr):
    denominator = arr.max() - arr.min()
    if denominator == 0:
        return arr
    return (arr - arr.min()) / denominator

## Load in DICOM
# z thickness is 3mm which is way bigger than what we want. Covid-19 dataset with 1.25mm thickness 
# would be much better if we can calibrate it properly (or clamp so min is -1000?).
arrs = []
for i in range(130):
    path = f"../data/manifest-OtXaMwL56190865641215613043/QIN LUNG CT/QIN-LSC-0003/08-06-2003-1-CT Thorax wContrast-41946/2.000000-THORAX W  3.0  B41 Soft Tissue-71225/1-{i+1:03}.dcm"
    ds = dcmread(path)
    arr = ds.pixel_array
    arrs.append(arr)

arr = np.array(arrs).astype(np.float32) # 130, 512, 512
arr = np.swapaxes(arr, 0, 1) # swap axes for nicer orientation
x_lim, y_lim, z_lim = 511, 129, 511 # replace with ct_shape or vice versa

## Extract parameters from metadata
# voxel size in mm
voxel_size = torch.tensor([float(ds.PixelSpacing[0]), float(ds.SliceThickness), float(ds.PixelSpacing[1])])
# HU rescaling params
rescale_intercept = float(ds.RescaleIntercept)
rescale_slope = float(ds.RescaleSlope)

## Rescale HU and calculate real world patient sizes in cm (attenuation coefficients are in cm^{-1})
# rescale to standard HU
arr = rescale_intercept + (arr * rescale_slope) 
# size of ct scan in cm (= voxel size * num voxels)
ct_shape = torch.tensor([x_lim, y_lim, z_lim])+1
ct_size = voxel_size * ct_shape / 10
ct_size = ct_size.to(device)

# nearest and furthest z values based on radius of source and size of ct scan
z_near = radius - (ct_size[-1].item() / 2)
z_far = radius + (ct_size[-1].item() / 2)

## Calculate x-ray source positions
_coord_from_blender = util.coord_from_blender()
render_poses = torch.stack(
    [
        _coord_from_blender @ util.pose_spherical(angle, elevation, radius)
        for angle in np.linspace(-180, 180, num_views + 1)[:-1]
    ],
    0,
)

## Wrapper to get closest CT voxel for any xyz coordinate
class CTImage(torch.nn.Module):
    def __init__(self, img, water_coeff=0.08):
        super().__init__()
        # Convert from HU to linear attenuation coefficients
        # Changing water attenuation coefficient changes contrast
        self.water_coeff = water_coeff
        self.img = ((img.clamp(min=-1000) / 1000) + 1) * water_coeff
    
    def forward(self, xyz, coarse=True, viewdirs=None, far=False):
        # xyz is in range -0.5*ct_size to 0.5*ct_size. Scale to be in range [0,1]
        xyz = xyz.squeeze(0)

        xyz = (xyz + (ct_size.unsqueeze(0) / 2)) / ct_size.unsqueeze(0)

        # scale xyz to nearest value in pixel space
        xyz[:,0] *= x_lim
        xyz[:,1] *= y_lim
        xyz[:,2] *= z_lim
        xyz = xyz.long().transpose(0,1) 

        # get rows where values are out of bounds and put them back in bounds
        mask = (xyz[0,:]<0) | (xyz[1,:]<0) | (xyz[2,:]<0) | (xyz[0,:]>x_lim) | (xyz[1,:]>y_lim) | (xyz[2,:]>z_lim)
        xyz[:,mask] = 0

        sigma = self.img[tuple(xyz)]
        # Anything out of bounds set as air
        sigma[mask] = 0
        sigma = sigma.reshape(1, -1, 1)
        rgb = torch.ones(1, sigma.size(1), 3).to(device)
        return torch.cat((rgb, sigma), dim=-1).to(device)


focal = torch.tensor(focal, dtype=torch.float32, device=device)

# TODO: Change num coarse and fine to take into account each voxel exactly once
image = CTImage(torch.tensor(arr).to(device))
renderer = NeRFRenderer(
    n_coarse=512, depth_std=0.01, sched=[], 
    white_bkgd=False, composite_x_ray=True, eval_batch_size=50000, lindisp=True
).to(device=device)
render_par = renderer.bind_parallel(image, [0], simple_output=True).eval()

render_rays = util.gen_rays_variable_sensor(render_poses, width_pixels, height_pixels, width, height, focal, z_near, z_far).to(device)

all_rgb_fine = []
for rays in tqdm(torch.split(render_rays.view(-1, 8), 80000, dim=0)):
    rgb, _depth = render_par(rays[None])
    all_rgb_fine.append(rgb[0])
_depth = None
rgb_fine = torch.cat(all_rgb_fine)

# rgb_fine = 1-normalize(rgb_fine)
rgb_fine = torch.clamp(1 - rgb_fine, 0, 1)

frames = (rgb_fine.view(num_views, H, W).cpu().numpy() * 255).astype(
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