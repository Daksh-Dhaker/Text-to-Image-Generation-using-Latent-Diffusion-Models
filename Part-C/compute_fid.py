import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID from two folders")
    parser.add_argument("--real_dir", required=True, help="Directory with real images")
    parser.add_argument("--gen_dir", required=True, help="Directory with generated/recon images")
    parser.add_argument("--device", default=None, help="Override device: cuda or cpu")
    parser.add_argument(
        "--output_file",
        default=None,
        help="Optional output file path (default: gen_dir/fid.txt)",
    )
    return parser.parse_args()


def ensure_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")


def main() -> None:
    args = parse_args()
    real_dir = Path(args.real_dir)
    gen_dir = Path(args.gen_dir)
    ensure_dir(real_dir, "Real image directory")
    ensure_dir(gen_dir, "Generated image directory")

    try:
        from cleanfid import fid
    except ImportError:
        print("clean-fid not installed. Run: pip install clean-fid")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    score = fid.compute_fid(str(real_dir), str(gen_dir), device=device)
    print(f"FID = {score:.4f}")

    output_path = Path(args.output_file) if args.output_file else (gen_dir / "fid.txt")
    output_path.write_text(f"FID: {score:.4f}\n")


if __name__ == "__main__":
    main()
