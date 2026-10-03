"""Run this inside Google Colab with a GPU runtime. Requires the NVlabs/stylegan3
repo cloned (see README.md 'StyleGAN3 generation') and a pretrained .pkl
checkpoint (stylegan3-r-ffhq-*.pkl or stylegan3-t-ffhq-*.pkl)."""
import argparse
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic StyleGAN3 face images for dataset augmentation."
    )
    parser.add_argument("--stylegan3-repo", type=Path, required=True, help="Path to cloned NVlabs/stylegan3 repo.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a stylegan3-r/t .pkl checkpoint.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--resolution", type=int, choices=[256, 1024], default=256)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=None, help="Save as JPEG at this quality instead of PNG, to match a JPEG training set's compression.")
    parser.add_argument("--no-custom-ops", action="store_true", help="Use StyleGAN3's pure-PyTorch reference ops instead of compiling CUDA plugins (for machines without nvcc).")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(args.stylegan3_repo))
    import numpy as np
    import torch
    import PIL.Image
    import dnnlib
    import legacy

    if args.no_custom_ops:
        from torch_utils.ops import bias_act, filtered_lrelu, upfirdn2d
        for op in (bias_act, filtered_lrelu, upfirdn2d):
            op._init = lambda: False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with dnnlib.util.open_url(str(args.checkpoint)) as f:
        generator = legacy.load_network_pkl(f)["G_ema"].to(device)

    for i in range(args.num_images):
        seed = args.seed_start + i
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, generator.z_dim)).to(device)
        with torch.no_grad():
            image = generator(z, None, truncation_psi=0.7, noise_mode="const")
        image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        if args.jpeg_quality is None:
            PIL.Image.fromarray(image, "RGB").save(args.out_dir / f"stylegan3_{seed:06d}.png")
        else:
            PIL.Image.fromarray(image, "RGB").save(args.out_dir / f"stylegan3_{seed:06d}.jpg", quality=args.jpeg_quality)


if __name__ == "__main__":
    main()
