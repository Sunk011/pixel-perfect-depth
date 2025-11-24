import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import torch.nn.functional as F
from ppd.utils.set_seed import set_seed
from ppd.models.ppd import PixelPerfectDepth

if __name__ == '__main__':
    set_seed(666) # set random seed
    parser = argparse.ArgumentParser(description='Pixel-Perfect Depth Visualization Only')
    parser.add_argument('--img_path', type=str, default='assets/examples')
    parser.add_argument('--input_size', type=int, default=[1024, 768])
    parser.add_argument('--outdir', type=str, default='depth_vis_only')
    parser.add_argument('--semantics_pth', type=str, default='checkpoints/depth_anything_v2_vitl.pth')
    parser.add_argument('--sampling_steps', type=int, default=4)

    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    model = PixelPerfectDepth(semantics_pth=args.semantics_pth, sampling_steps=args.sampling_steps)
    model.load_state_dict(torch.load('checkpoints/ppd.pth', map_location='cpu'), strict=False)

    model = model.to(DEVICE).eval()

    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
        filenames = sorted(filenames)

    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral')

    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        try:
            image = cv2.imread(filename)
            if image is None:
                print(f"Skipping {filename}, not a valid image.")
                continue

            H, W = image.shape[:2]
            depth, _ = model.infer_image(image)
            depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)[0, 0]
            depth = depth.squeeze().cpu().numpy()
            
            vis_depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            vis_depth = vis_depth.astype(np.uint8)
            vis_depth = (cmap(vis_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            save_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
            cv2.imwrite(save_path, vis_depth)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
