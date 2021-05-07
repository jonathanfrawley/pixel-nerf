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


# hyperparams
elevation = 0.0
radius = 1.3
z_near, z_far = 0.8, 1.8
num_views = 100

arr_max = 1.0
arr_min = 0.7832
denominator = arr_max - arr_min

gif = True
resume = True
device = 'cuda'
output = os.path.join(ROOT_DIR, "output")

arrs = []
for i in range(130):
    path = f"../data/manifest-OtXaMwL56190865641215613043/QIN LUNG CT/QIN-LSC-0003/08-06-2003-1-CT Thorax wContrast-41946/2.000000-THORAX W  3.0  B41 Soft Tissue-71225/1-{i+1:03}.dcm"
    ds = dcmread(path)
    arr = ds.pixel_array
    arrs.append(arr)

arr = np.array(arrs).astype(np.float32) # 216, 512, 512
arr = arr - 1024 # take into account rescale intercept
arr = np.swapaxes(arr, 0, 1)
x_lim, y_lim, z_lim = 511, 129, 511
# arr is in range [-1024, 3071] 
# NOTE very large range for nn to approximate

_coord_to_blender = util.coord_to_blender()
_coord_from_blender = util.coord_from_blender()

class CTImage(torch.nn.Module):
    def __init__(self, img, water_coeff=0.2):
        super().__init__()
        # Convert from HU to linear attenuation coefficients
        self.water_coeff = water_coeff
        self.img = ((img / 1000) + 1) * water_coeff
    
    def forward(self, xyz, coarse=True, viewdirs=None, far=False):
        # xyz appears to be roughly in the range [-1.7, 1.7], not sure why
        # Scaling so projected images show the whole body.
        xyz = xyz.squeeze(0)
        xyz = (xyz + 1) / 2

        xyz[:,0] *= x_lim
        xyz[:,1] *= y_lim
        xyz[:,2] *= z_lim
        xyz = xyz.long().transpose(0,1)

        # get rows where values are out of bounds and put them back in bounds
        mask = (xyz[0,:]<0) | (xyz[1,:]<0) | (xyz[2,:]<0) | (xyz[0,:]>x_lim) | (xyz[1,:]>y_lim) | (xyz[2,:]>z_lim)
        xyz[:,mask] = 0

        sigma = self.img[tuple(xyz)]
        # Anything out of bounds set back to air
        sigma[mask] = ((-1024 / 1000) + 1) * self.water_coeff
        sigma = sigma.reshape(1, -1, 1)
        rgb = torch.ones(sigma.size(0), sigma.size(1), 3).to(device)
        return torch.cat((rgb, sigma), dim=-1).to(device)

render_poses = torch.stack(
    [
        _coord_from_blender @ util.pose_spherical(angle, elevation, radius)
        for angle in np.linspace(-180, 180, num_views + 1)[:-1]
    ],
    0,
)

H = W = 128
focal = torch.tensor(50.0, dtype=torch.float32, device=device) # 512: 131.25, 256: 80.0, 128: 50.0

# Render training data or load in if already rendered
if os.path.exists(os.path.join(output, f'training_ct_{H}.pkl')):
    ct_gt = torch.load(os.path.join(output, f'training_ct_{H}.pkl'))
else:
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

    # Using Beer's law: I=I_0 exp(-int_0^D \mu(x)dx) where \mu is the linear attenuation 
    # coefficient of the material related to the HU number
    # not true integral since distance isn't in cm
    rgb_fine = torch.exp(-rgb_fine)

    # normalize
    rgb_fine = (rgb_fine - rgb_fine.min()) / (rgb_fine.max() - rgb_fine.min())
    rgb_fine = 1-rgb_fine
    ct_gt = rgb_fine.view(num_views, H, W, 3)

    torch.save(ct_gt, os.path.join(output, f'training_ct_{H}.pkl'))


render_rays = util.gen_rays(render_poses, W, H, focal, z_near, z_far).to(device=device)

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
        out = self.NeRF(xyz, coarse=coarse, viewdirs=viewdirs, far=far)
        # Ideally the NeRF should only output densities (linear attenuation coefficients)
        # Note that these can be negative whereas the currently architecture has a relu at the end
        # For now just use a full NeRF for simplicity
        # ones = torch.ones_like(out)
        # return torch.cat([ones]*3 + [out[:,:,-1:]], dim=-1)
        return out

conf = ConfigFactory.parse_file('conf/exp/ct_single.conf')
net = NeRFDepthOnly(conf).to(device) 

if resume:
    print("Resuming from file " + os.path.join(output, "single_ct_nerf.pkl"))
    net = torch.load(os.path.join(output, "single_ct_nerf.pkl"))

renderer_net = NeRFRenderer(
    n_coarse=64, n_fine=32, n_fine_depth=16, depth_std=0.01, sched=[], white_bkgd=False
).to(device=device)
render_par_net = renderer_net.bind_parallel(net, [0], simple_output=False).eval()


## Training loop
ray_batch_size = 128
mseloss = torch.nn.MSELoss(reduction="mean")
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

for step in range(1000000):
    pix_inds = torch.randint(0, num_views * H * W, (ray_batch_size,))

    img_gt = ct_gt.contiguous().reshape(-1, 3)[pix_inds].unsqueeze(0)
    rays = render_rays.view(-1, 8)[pix_inds].unsqueeze(0)

    render_dict = DotMap(render_par_net(rays, want_weights=True,))
    coarse = render_dict.coarse
    fine = render_dict.fine

    # rgb_loss = mseloss(coarse.rgb, img_gt)
    # fine_loss = mseloss(fine.rgb, img_gt)
    rgb_loss = torch.abs(coarse.rgb - img_gt).mean()
    fine_loss = torch.abs(fine.rgb - img_gt).mean()
    rgb_loss = rgb_loss + fine_loss

    loss = rgb_loss
    optim.zero_grad()
    loss.backward()
    optim.step()

    if step % 100 == 0:
        print(f"Step: {step}, Loss: {loss:.3f}, Fine Loss: {fine_loss:.3f}")
    
    if step % 1000 == 0:
        torch.save(net, os.path.join(output, "single_ct_nerf.pkl"))
    
    # make video
    if step % 10000 == 0 and step > 0:
        num_eval_views = 100
        with torch.no_grad():
            render_poses_eval = torch.stack(
                [
                    _coord_from_blender @ util.pose_spherical(angle, elevation, radius)
                    for angle in np.linspace(-180, 180, num_eval_views + 1)[:-1]
                ],
                0,
            )
            render_rays_eval = util.gen_rays(render_poses_eval, W, H, focal, z_near, z_far).to(device=device)
            all_rgb_fine = []
            for rays in tqdm(torch.split(render_rays_eval.view(-1, 8), 80000, dim=0)):
                render_dict = DotMap(render_par_net(rays[None]))
                rgb = render_dict.fine.rgb
                all_rgb_fine.append(rgb[0])
            _depth = None
            rgb_fine = torch.cat(all_rgb_fine).clamp(0,1)
            rgb_fine = rgb_fine.view(num_eval_views, H, W, 3)
            frames = (rgb_fine.view(num_eval_views, H, W, 3).cpu().numpy() * 255).astype(np.uint8)
            if gif:
                vid_path = os.path.join(output, f"raw_data_{step}_vid.gif")
                imageio.mimwrite(vid_path, frames, fps=24)
            else:
                vid_path = os.path.join(output, f"raw_data_{step}_vid.mp4")
                imageio.mimwrite(vid_path, frames, fps=24, quality=8)



