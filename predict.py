#!/usr/bin/env python3
"""
CLI Diagnostic Prediction Tool for CAD-Analysis-Dashboard.
Usage:
    python predict.py --image path/to/ct_scan.jpg
    python predict.py --dir path/to/folder/ --output results/
"""

import argparse
import sys
import re
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
        sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
    except Exception:
        pass

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.inference import diagnose_image, create_cine_gif


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def process_single_image(image_path: Path, output_dir: Path = None):
    print(f"\n[+] Processing image: {image_path.name}...")
    try:
        res = diagnose_image(image_path, filename=image_path.name)
    except Exception as e:
        print(f"[!] Error processing {image_path.name}: {e}")
        return None, None

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

    row = {
        "Image": image_path.name,
        "Diagnosis": pred,
        "Confidence_Percent": conf,
        "COVID_Probability": p_covid,
        "Normal_Probability": p_normal
    }
    return row, res


def main():
    parser = argparse.ArgumentParser(description="CAD Radiomics Clinical Image Diagnosis Tool")
    parser.add_argument("--image", "-i", type=str, help="Path to single CT/Chest X-ray image (DICOM, JPG, PNG)")
    parser.add_argument("--dir", "-d", type=str, help="Path to directory containing CT slices for batch diagnosis")
    parser.add_argument("--output", "-o", type=str, default="output", help="Directory to save diagnosis outputs (default: output/)")
    args = parser.parse_args()

    if not args.image and not args.dir:
        print("[!] Please provide either --image or --dir. Use -h for help.")
        sys.exit(1)

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        img_path = Path(args.image).resolve()
        if not img_path.is_file():
            print(f"[!] Image file not found: {img_path}")
            sys.exit(1)
        row, _ = process_single_image(img_path, output_dir)
        print(f"\n[+] Outputs successfully saved to: {output_dir}")

    elif args.dir:
        dir_path = Path(args.dir).resolve()
        if not dir_path.is_dir():
            print(f"[!] Directory not found: {dir_path}")
            sys.exit(1)

        valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".dcm", ".dicom"}
        image_files = [p for p in dir_path.iterdir() if p.suffix.lower() in valid_extensions]

        if not image_files:
            print(f"[!] No valid image files found in {dir_path}")
            sys.exit(1)

        # Natural sort order
        image_files = sorted(image_files, key=natural_sort_key)
        print(f"[+] Found {len(image_files)} sequential CT images to diagnose in {dir_path}")

        results_rows = []
        raw_results = []
        for img_p in image_files:
            row, res = process_single_image(img_p, output_dir)
            if row and res:
                results_rows.append(row)
                raw_results.append(res)

        if results_rows:
            # 1. Save summary CSV
            df_results = pd.DataFrame(results_rows)
            csv_path = output_dir / "batch_diagnosis_summary.csv"
            df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"\n[+] Batch summary saved to: {csv_path}")

            # 2. Save Cine-Loop GIF
            gif_path = output_dir / "patient_cineloop.gif"
            gif_bytes = create_cine_gif(raw_results, duration_ms=300, side_by_side=True)
            gif_path.write_bytes(gif_bytes)
            print(f"[+] Cine-Loop GIF animation saved to: {gif_path}")

            # 3. Clinical Conclusion
            total_slices = len(raw_results)
            normal_cnt = sum(1 for r in raw_results if r["prediction"] == "Normal")
            covid_cnt = sum(1 for r in raw_results if r["prediction"] == "COVID-19")
            avg_norm = float(np.mean([r["probability_normal"] for r in raw_results])) * 100
            avg_covid = float(np.mean([r["probability_covid"] for r in raw_results])) * 100

            if covid_cnt == 0:
                conclusion_text = f"Kết luận lâm sàng: Cả {total_slices} lát cắt của bệnh nhân đều được chẩn đoán là NORMAL (Âm tính với tổn thương đông đặc/kính mờ dạng COVID-19), với xác suất trung bình {avg_norm:.1f}%."
            else:
                conclusion_text = f"Kết luận lâm sàng: Phát hiện {covid_cnt}/{total_slices} lát cắt có dấu hiệu tổn thương nghi ngờ COVID-19 (Đông đặc/kính mờ), với xác suất trung bình {avg_covid:.1f}%. Cần bác sĩ hội chẩn lâm sàng chi tiết!"

            print("\n" + "=" * 70)
            print(f"  {conclusion_text}")
            print("=" * 70)

            # 4. Save clinical report
            report_path = output_dir / "clinical_report.txt"
            report_path.write_text(conclusion_text + f"\nSummary:\n- Total: {total_slices}\n- Normal: {normal_cnt}\n- COVID-19: {covid_cnt}\n", encoding="utf-8")
            print(f"[+] Clinical report saved to: {report_path}")


if __name__ == "__main__":
    main()
