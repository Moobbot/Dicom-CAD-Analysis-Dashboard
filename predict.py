#!/usr/bin/env python3
"""
CLI Diagnostic Prediction Tool for CAD-Analysis-Dashboard.
Usage:
    python predict.py --image path/to/ct_scan.jpg
    python predict.py --dir path/to/folder/ --output results/
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
import pandas as pd

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.inference import diagnose_image


def process_single_image(image_path: Path, output_dir: Path = None):
    print(f"\n[+] Processing image: {image_path.name}...")
    try:
        res = diagnose_image(image_path)
    except Exception as e:
        print(f"[!] Error processing {image_path.name}: {e}")
        return None

    pred = res["prediction"]
    conf = res["confidence"]
    p_covid = res["probability_covid"] * 100
    p_normal = res["probability_normal"] * 100

    print("=" * 55)
    print(f"  DIAGNOSIS RESULT  : {pred.upper()}")
    print(f"  CONFIDENCE        : {conf:.2f}%")
    print(f"  COVID-19 Prob     : {p_covid:.2f}%")
    print(f"  Normal Prob       : {p_normal:.2f}%")
    print("=" * 55)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        # Save segmented mask
        mask_path = output_dir / f"{stem}_mask.png"
        Image.fromarray(res["mask"]).save(mask_path)

        # Save visual overlay
        overlay_path = output_dir / f"{stem}_overlay.png"
        Image.fromarray(res["overlay"]).save(overlay_path)

        print(f"[i] Saved lung mask: {mask_path}")
        print(f"[i] Saved overlay  : {overlay_path}")

    return {
        "Image": image_path.name,
        "Diagnosis": pred,
        "Confidence_Percent": conf,
        "COVID_Probability": p_covid,
        "Normal_Probability": p_normal
    }


def main():
    parser = argparse.ArgumentParser(description="CAD Radiomics Clinical Image Diagnosis Tool")
    parser.add_argument("--image", "-i", type=str, help="Path to single CT/Chest X-ray image")
    parser.add_argument("--dir", "-d", type=str, help="Path to directory containing images for batch diagnosis")
    parser.add_argument("--output", "-o", type=str, default="diagnosis_results", help="Directory to save diagnosis outputs")
    args = parser.parse_args()

    if not args.image and not args.dir:
        print("[!] Please provide either --image or --dir. Use -h for help.")
        sys.exit(1)

    output_dir = Path(args.output)

    if args.image:
        img_path = Path(args.image)
        if not img_path.is_file():
            print(f"[!] Image file not found: {img_path}")
            sys.exit(1)
        process_single_image(img_path, output_dir)

    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"[!] Directory not found: {dir_path}")
            sys.exit(1)

        valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        image_files = [p for p in dir_path.iterdir() if p.suffix.lower() in valid_extensions]

        if not image_files:
            print(f"[!] No valid image files found in {dir_path}")
            sys.exit(1)

        print(f"[+] Found {len(image_files)} images to diagnose in {dir_path}")
        results = []
        for img_p in image_files:
            row = process_single_image(img_p, output_dir)
            if row:
                results.append(row)

        if results:
            df_results = pd.DataFrame(results)
            csv_path = output_dir / "batch_diagnosis_summary.csv"
            df_results.to_csv(csv_path, index=False)
            print(f"\n[+] Batch summary saved to: {csv_path}")


if __name__ == "__main__":
    main()
