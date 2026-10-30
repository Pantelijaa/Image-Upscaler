"""
Download and prepare the standard super-resolution datasets.

Training set (data/train/HR, 291 images, as used by VDSR):
    T91     - 91 images from Yang et al. (mirrored in the DCSCN repository)
    BSD200  - the 200 training images of BSDS300

Test sets (data/test/<Set>/HR and data/test/<Set>/LR_x<scale>):
    Set5, Set14, BSD100, Urban100  - HR images

Protocol
    * HR images are converted to RGB PNG and mod-cropped so both width and height
      are divisible by every scale factor in SCALES.
    * LR images are produced from the cropped HR with bicubic downscaling
      (Pillow, antialiased - the closest readily available match to MATLAB's
      imresize) and saved as PNG.
    * File names are identical in HR/ and LR_x<scale>/ so the C++ benchmark can
      pair them directly.

Usage:
    python -I CNN/prepare_data.py            # everything
    python -I CNN/prepare_data.py --force    # regenerate even if outputs exist
    python -I CNN/prepare_data.py --skip-train
"""

import argparse
import io
import json
import math
import shutil
import sys
import tarfile
import urllib.request
from functools import reduce
from pathlib import Path

from PIL import Image

SCALES = [2, 4]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOWNLOADS = DATA / "_downloads"
TRAIN_DIR = DATA / "train" / "HR"
TEST_DIR = DATA / "test"

HF = "https://huggingface.co/datasets/eugenesiow/{set}/resolve/main/data/{set}_HR.tar.gz"
TEST_SETS = {
	"Set5": HF.format(set="Set5"),
	"Set14": HF.format(set="Set14"),
	"BSD100": HF.format(set="BSD100"),
	"Urban100": HF.format(set="Urban100"),
}
BSDS300_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
T91_API = "https://api.github.com/repos/jiny2001/dcscn-super-resolution/contents/data/yang91"


def log(msg):
	print(msg, flush=True)


def download(url, dest):
	"""Download url to dest unless it is already there."""
	if dest.exists() and dest.stat().st_size > 0:
		log(f"  cached   {dest.name}")
		return dest
	dest.parent.mkdir(parents=True, exist_ok=True)
	log(f"  download {url}")
	req = urllib.request.Request(url, headers={"User-Agent": "image-upscaler-prepare-data"})
	with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
		shutil.copyfileobj(resp, f)
	log(f"           -> {dest.stat().st_size / 1e6:.1f} MB")
	return dest


def extract(archive, dest):
	"""Extract a .tar.gz/.tgz archive into dest (once)."""
	marker = dest / ".extracted"
	if marker.exists():
		return dest
	dest.mkdir(parents=True, exist_ok=True)
	log(f"  extract  {archive.name}")
	with tarfile.open(archive, "r:*") as tar:
		tar.extractall(dest, filter="data")
	marker.touch()
	return dest


def list_images(directory):
	return sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def mod_crop(img, multiple):
	w, h = img.size
	return img.crop((0, 0, w - w % multiple, h - h % multiple))


def prepare_test_set(name, images, force):
	hr_dir = TEST_DIR / name / "HR"
	lr_dirs = {s: TEST_DIR / name / f"LR_x{s}" for s in SCALES}
	if hr_dir.exists() and not force and len(list_images(hr_dir)) == len(images):
		log(f"  ready    {name} ({len(images)} images)")
		return
	hr_dir.mkdir(parents=True, exist_ok=True)
	for d in lr_dirs.values():
		d.mkdir(parents=True, exist_ok=True)

	multiple = reduce(math.lcm, SCALES)
	for src in images:
		stem = src.stem
		hr = mod_crop(Image.open(src).convert("RGB"), multiple)
		hr.save(hr_dir / f"{stem}.png")
		w, h = hr.size
		for s, d in lr_dirs.items():
			hr.resize((w // s, h // s), Image.BICUBIC).save(d / f"{stem}.png")
	log(f"  wrote    {name}: {len(images)} HR images, LR at x{', x'.join(map(str, SCALES))}")


def prepare_test_sets(force):
	log("Test sets")
	for name, url in TEST_SETS.items():
		dl = DOWNLOADS / name
		archive = download(url, dl / Path(url).name)
		extracted = extract(archive, dl / "extract")
		prepare_test_set(name, list_images(extracted), force)


def download_t91():
	dl = DOWNLOADS / "T91"
	listing = dl / "listing.json"
	if not listing.exists():
		download(T91_API, listing)
	entries = json.loads(listing.read_text(encoding="utf-8"))
	files = []
	for e in entries:
		if e.get("type") == "file" and Path(e["name"]).suffix.lower() in IMAGE_EXTS:
			files.append(download(e["download_url"], dl / "images" / e["name"]))
	return files


def prepare_train_set(force):
	log("Training set")
	if TRAIN_DIR.exists() and not force and len(list_images(TRAIN_DIR)) == 291:
		log("  ready    data/train/HR (291 images)")
		return
	if TRAIN_DIR.exists():
		shutil.rmtree(TRAIN_DIR)
	TRAIN_DIR.mkdir(parents=True)

	t91 = download_t91()
	for src in t91:
		Image.open(src).convert("RGB").save(TRAIN_DIR / f"t91_{src.stem}.png")
	log(f"  wrote    T91: {len(t91)} images")

	dl = DOWNLOADS / "BSDS300"
	archive = download(BSDS300_URL, dl / Path(BSDS300_URL).name)
	extracted = extract(archive, dl / "extract")
	train_imgs = [p for p in list_images(extracted) if p.parent.name == "train"]
	for src in train_imgs:
		Image.open(src).convert("RGB").save(TRAIN_DIR / f"bsd200_{src.stem}.png")
	log(f"  wrote    BSD200: {len(train_imgs)} images")


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--force", action="store_true", help="regenerate outputs even if present")
	parser.add_argument("--skip-train", action="store_true", help="only prepare the test sets")
	parser.add_argument("--skip-test", action="store_true", help="only prepare the training set")
	args = parser.parse_args()

	if not args.skip_test:
		prepare_test_sets(args.force)
	if not args.skip_train:
		prepare_train_set(args.force)
	log("Done.")


if __name__ == "__main__":
	main()
