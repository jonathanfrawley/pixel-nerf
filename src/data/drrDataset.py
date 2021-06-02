import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
import cv2
from util import get_image_to_tensor_balanced, get_mask_to_tensor


class drrDataset(torch.utils.data.Dataset):
    """
    Dataset of DRR generated from CT scans
    """

    def __init__(
        self,
        path,
        stage="train",
        list_prefix="softras_",
        image_size=None,
        scale_focal=False,
        max_imgs=100000,
        z_near=0.0,
        z_far=50.0,
        skip_step=None,
    ):
        """
        :param path dataset root path
        :param image_size result image size (resizes if different); None to keep original size
        :param scale_focal if true, assume focal length is specified for
        image of side length 2 instead of actual image size. This is used
        where image coordinates are placed in [-1, 1].
        """
        super().__init__()
        self.base_path = path
        assert os.path.exists(self.base_path)

        # transforms for images
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()
        self.image_size = image_size
        
        self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        self._coord_trans_cam = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        
        self.scale_focal = scale_focal
        self.max_imgs = max_imgs
        self.z_near = z_near
        self.z_far = z_far
        self.lindisp = False
    
        self.rgb_paths = []
        self.project_mats = []
        files = os.listdir(self.base_path)
        for f in files:
            if f.endswith(".png"):
                self.rgb_paths.append(os.path.join(self.base_path, f))
            elif f.endswith(".txt"):
                self.project_mats.append(os.path.join(self.base_path, f))
        self.rgb_paths.sort()
        self.project_mats.sort()

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, index):
        all_imgs = []
        all_masks = []
        all_poses = []
        all_bboxes = []
        focal = None

        for idx in range(len(self.rgb_paths)):
            im = imageio.imread(self.rgb_paths[idx])
#            img = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)[..., :3]
            img = np.repeat(im[..., np.newaxis], 3, -1)[..., :3]
            mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 195
            mask_tensor = self.mask_to_tensor(mask)
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                raise RuntimeError(
                    "ERROR: Bad image at", self.rgb_paths[idx], "please investigate!"
                )
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            P = np.linalg.inv(np.vstack((np.genfromtxt(self.project_mats[idx], dtype=float,
                              skip_header=1, skip_footer=12), np.array([0.0,0.0,0.0,1.0]))))
            P = P[:3]

            K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
            K = K / K[2, 2]

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R.transpose()
            pose[:3, 3] = (-R @ (t[:3] / t[3]))[:, 0]

            c = focal = torch.from_numpy(np.genfromtxt(self.project_mats[0],
                                               dtype=np.float32, skip_header=0,
                                               skip_footer=15))

            pose = (
                self._coord_trans_world
                @ torch.tensor(pose, dtype=torch.float32)
                @ self._coord_trans_cam
            )

            img_tensor = self.image_to_tensor(img)
            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_bboxes.append(bbox)
            all_poses.append(pose)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)

        if self.image_size is not None and all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        result = {
            "path": self.base_path,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
            "masks": all_masks,
            "bbox": all_bboxes,
            "c": c,
        }
        return result
