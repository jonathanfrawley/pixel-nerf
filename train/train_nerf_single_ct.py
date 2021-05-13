import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

import util
import torch
import numpy as np
from model import make_model
from render import NeRFRenderer
import tqdm
import imageio

from dotmap import DotMap
from pyhocon import ConfigFactory

from pydicom import dcmread

from tqdm import tqdm

def normalize(arr):
    denominator = arr.max() - arr.min()
    if denominator == 0:
        return arr
    return (arr - arr.min()) / denominator

## Hyperparams
elevation = 0.0
num_views = 100
## Radius and focal length set as in 2.4.f as here https://iopscience.iop.org/article/10.1088/0031-9155/45/10/305/pdf
radius = 100 # how far away the x-ray source is from centre of the the patient in cm
focal = 140 # how far away the x-ray source is from the detector in cm
## Resolution and sensor size can be set independently
W = H = width_pixels = height_pixels = 128 # number of pixels over width/height
width = height = 60 # width/height of detector in cm

gif = True
resume = False
device = 'cuda'
output = os.path.join(ROOT_DIR, "output")

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
ct_shape = torch.tensor([x_lim, y_lim, z_lim]) + 1
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

# Render training data or load in if already rendered
# if os.path.exists(os.path.join(output, f'training_ct_{H}.pkl')):
#     ct_gt = torch.load(os.path.join(output, f'training_ct_{H}.pkl'))
# else:
image = CTImage(torch.tensor(arr).to(device))
renderer = NeRFRenderer(
    n_coarse=512, depth_std=0.01, sched=[], 
    white_bkgd=False, composite_x_ray=False, eval_batch_size=50000, lindisp=True
).to(device=device)
render_par = renderer.bind_parallel(image, [0], simple_output=True).eval()

render_rays = util.gen_rays_variable_sensor(render_poses, width_pixels, height_pixels, width, height, focal, z_near, z_far).to(device)

all_rgb_fine = []
for rays in tqdm(torch.split(render_rays.view(-1, 8), 80000, dim=0)):
    rgb, _depth = render_par(rays[None])
    all_rgb_fine.append(rgb[0])
_depth = None
rgb_fine = torch.cat(all_rgb_fine)

# rgb_fine = torch.clamp(1 - rgb_fine, 0, 1)
ct_gt_min = rgb_fine.min()
ct_gt_max = rgb_fine.max()
rgb_fine = (rgb_fine - ct_gt_min) / (ct_gt_max - ct_gt_min)
ct_gt = rgb_fine.view(num_views, H, W, 3)

torch.save(ct_gt, os.path.join(output, f'training_ct_{H}.pkl'))

render_rays = util.gen_rays_variable_sensor(render_poses, width_pixels, height_pixels, width, height, focal, z_near, z_far).to(device)

frames = (ct_gt.view(num_views, H, W, 3).cpu().numpy() * 255).astype(
    np.uint8
)

# Write training data to file to check it's correct
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


class NeRFDepthOnly(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.NeRF = make_model(conf["model"])
    
    def forward(self, xyz, coarse=True, viewdirs=None, far=False):
        # rescale xyz to between -1 and 1
        xyz = xyz / (ct_size.reshape(1,1,-1)/2)
        xyz = xyz / 1.7 # because we extend beyond the size of the CT. More exact would be better
        out = self.NeRF(xyz, coarse=coarse, viewdirs=viewdirs, far=far)
        # Ideally the NeRF should only output densities (linear attenuation coefficients)
        # Note that these can be negative whereas the currently architecture has a relu at the end
        # For now just use a full NeRF for simplicity

        # ones = torch.ones_like(out[...,:3])
        # return torch.cat([ones, out[:,:,-1:]], dim=-1)
        return out

conf = ConfigFactory.parse_file('conf/exp/ct_single.conf')
net = NeRFDepthOnly(conf).to(device) 

if resume:
    print("Resuming from file " + os.path.join(output, "single_ct_nerf.pkl"))
    net = torch.load(os.path.join(output, "single_ct_nerf.pkl"))

# Unlike typical 3D models where there is lots of empty space, the bulk of a CT scan is
# full on non-zero contents so the fine approach doesn't work so well.
# NOTE: composite_x_ray doesn't seem to work in this case, it is definitely still broken.
# TODO: A better compositing system which does the integration properly by working out how far
#       the ray passes through each voxel rather than using loads of steps would be better
renderer_net = NeRFRenderer(
    n_coarse=512, n_fine=0, n_fine_depth=0, depth_std=0.01, sched=[], white_bkgd=False,
    composite_x_ray=False, lindisp=True
).to(device=device)
render_par_net = renderer_net.bind_parallel(net, [0], simple_output=False).eval()


## Training loop
ray_batch_size = 128
mseloss = torch.nn.MSELoss(reduction="mean")
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

losses = np.array([])
for step in range(1000000):
    pix_inds = torch.randint(0, num_views * H * W, (ray_batch_size,))

    img_gt = ct_gt.contiguous().reshape(-1, 3)[pix_inds].unsqueeze(0)
    rays = render_rays.view(-1, 8)[pix_inds].unsqueeze(0)

    render_dict = DotMap(render_par_net(rays, want_weights=True,))
    coarse = render_dict.coarse
    fine = render_dict.fine

    out = (coarse.rgb - ct_gt_min) / (ct_gt_max - ct_gt_min)

    # coarse_loss = mseloss(coarse.rgb.squeeze(-1), 1-img_gt)
    # fine_loss = mseloss(fine.rgb.squeeze(-1), 1-img_gt)
    # print(coarse.rgb.shape, img_gt.shape)
    coarse_loss = torch.abs(out - img_gt).mean() # why are they exactly the same?
    # fine_loss = torch.abs(fine.rgb - img_gt).mean()
    rgb_loss = coarse_loss #+ fine_loss

    loss = rgb_loss
    optim.zero_grad()
    loss.backward()
    optim.step()

    losses = np.append(losses, loss.item())

    if step % 100 == 0:
        print(f"Step: {step}, Loss: {losses.mean():.5f}")
        losses = np.array([])
        # print(f"Step: {step}, Loss: {loss:.3f}, Coarse Loss: {coarse_loss:.3f}, Fine Loss: {fine_loss:.3f}")
    
    if step % 1000 == 0:
        torch.save(net, os.path.join(output, "single_ct_nerf.pkl"))
    
    # make video
    if step % 10000 == 0 and step > 0:
        num_views_dict = {128: 100, 256: 25}
        fps_dict = {128: 24, 256: 12}
        num_eval_views = num_views_dict[W]
        fps = fps_dict[W]
        with torch.no_grad():
            render_poses_eval = torch.stack(
                [
                    _coord_from_blender @ util.pose_spherical(angle, elevation, radius)
                    for angle in np.linspace(-180, 180, num_eval_views + 1)[:-1]
                ],
                0,
            )
            render_rays_eval = util.gen_rays_variable_sensor(render_poses_eval, width_pixels, height_pixels, width, height, focal, z_near, z_far).to(device)
            all_rgb_fine = []
            for rays in tqdm(torch.split(render_rays_eval.view(-1, 8), 80000, dim=0)):
                render_dict = DotMap(render_par_net(rays[None]))
                rgb = render_dict.coarse.rgb
                all_rgb_fine.append(rgb[0])
            _depth = None
            rgb_fine = torch.cat(all_rgb_fine)
            rgb_fine = (rgb_fine - ct_gt_min) / (ct_gt_max - ct_gt_min)
            rgb_fine = rgb_fine.view(num_eval_views, H, W, 3)
            frames = (rgb_fine.view(num_eval_views, H, W, 3).cpu().numpy() * 255).astype(np.uint8)
            if gif:
                vid_path = os.path.join(output, f"raw_data_{step}_vid.gif")
                imageio.mimwrite(vid_path, frames, fps=fps)
            else:
                vid_path = os.path.join(output, f"raw_data_{step}_vid.mp4")
                imageio.mimwrite(vid_path, frames, fps=fps, quality=8)



