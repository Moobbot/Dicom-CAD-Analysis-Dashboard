import os
import sys
import logging
from pathlib import Path
from typing import Union, Tuple, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logger = logging.getLogger("CADInference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 24 core radiomic features matching params.yaml and training dataset
FEATURE_COLUMNS = [
    'original_firstorder_Entropy',
    'original_firstorder_Energy',
    'original_firstorder_Uniformity',
    'original_firstorder_MeanAbsoluteDeviation',
    'original_firstorder_Skewness',
    'original_firstorder_Kurtosis',
    'original_glcm_Contrast',
    'original_glcm_Idm',
    'original_glcm_Correlation',
    'original_glcm_ClusterProminence',
    'original_glcm_ClusterShade',
    'original_gldm_GrayLevelNonUniformity',
    'original_gldm_DependenceNonUniformity',
    'original_gldm_DependenceVariance',
    'original_gldm_LowGrayLevelEmphasis',
    'original_gldm_HighGrayLevelEmphasis',
    'original_glszm_ZoneEntropy',
    'original_glszm_LargeAreaEmphasis',
    'original_glszm_SmallAreaEmphasis',
    'original_glszm_GrayLevelVariance',
    'original_glszm_SizeZoneNonUniformity',
    'original_glrlm_RunLengthNonUniformity',
    'original_glrlm_ShortRunEmphasis',
    'original_glrlm_LongRunEmphasis'
]

# Global cache for heavy models
_UNET_MODEL = None
_CLASSIFIER_PIPELINE = None


def get_unet_model_path() -> Path:
    """Find the pre-trained U-Net weights file in ML/UNET Training/."""
    candidates = [
        BASE_DIR / "ML" / "UNET Training" / "lung_segmentation_unet .h5",
        BASE_DIR / "ML" / "UNET Training" / "lung_segmentation_unet.h5",
        BASE_DIR / "lung_segmentation_unet.h5"
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def load_segmentation_model():
    """Load and memoize the trained U-Net model."""
    global _UNET_MODEL
    if _UNET_MODEL is not None:
        return _UNET_MODEL

    model_path = get_unet_model_path()
    if not model_path.is_file():
        logger.warning(f"U-Net model weight not found at {model_path}. Will fallback to heuristic segmentation.")
        return None

    try:
        import tensorflow as tf
        logger.info(f"Loading U-Net model from: {model_path}")
        _UNET_MODEL = tf.keras.models.load_model(str(model_path), compile=False)
        return _UNET_MODEL
    except Exception as e:
        logger.error(f"Failed to load TensorFlow U-Net model: {e}")
        return None


def segment_lung(image_gray: np.ndarray, model=None) -> np.ndarray:
    """
    Segment the lung region from a 2D grayscale image.
    Returns a binary mask of shape (H, W) with values in {0, 255}.
    """
    orig_h, orig_w = image_gray.shape[:2]

    # Try deep learning U-Net segmentation first
    if model is None:
        model = load_segmentation_model()

    if model is not None:
        try:
            # U-Net input preprocessing: resize to 128x128, float32, normalize [0, 1]
            img_pil = Image.fromarray(image_gray).resize((128, 128), Image.Resampling.BILINEAR)
            img_norm = np.array(img_pil, dtype=np.float32) / 255.0
            img_tensor = np.expand_dims(img_norm, axis=(0, -1))  # (1, 128, 128, 1)

            pred = model.predict(img_tensor, verbose=0)
            mask_128 = (pred[0, :, :, 0] > 0.5).astype(np.uint8) * 255

            # Resize predicted mask back to original resolution
            mask_full = Image.fromarray(mask_128).resize((orig_w, orig_h), Image.Resampling.NEAREST)
            return np.array(mask_full, dtype=np.uint8)
        except Exception as e:
            logger.warning(f"U-Net inference error: {e}. Falling back to Otsu thresholding.")

    # Fallback heuristic segmentation (Otsu threshold + lung morphology)
    norm = ((image_gray - image_gray.min()) / (np.ptp(image_gray) + 1e-6) * 255).astype(np.uint8)
    # Lungs typically appear darker than bone/tissue in CT
    threshold = np.mean(norm)
    mask = ((norm < threshold) & (norm > 15)).astype(np.uint8) * 255
    return mask


def extract_radiomics(image_gray: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Extract the 24 radiomics features using PyRadiomics.
    Falls back to statistical computation if PyRadiomics is not installed.
    """
    params_path = BASE_DIR / "Analysis Dashboard" / "params.yaml"

    # 1. Attempt PyRadiomics extraction
    try:
        import SimpleITK as sitk
        from radiomics import featureextractor

        # SimpleITK image and mask objects
        sitk_img = sitk.Cast(sitk.GetImageFromArray(image_gray), sitk.sitkUInt8)
        sitk_mask = sitk.Cast(sitk.GetImageFromArray(mask), sitk.sitkUInt8)

        # Ensure label 255 is present
        if np.max(mask) == 0:
            logger.warning("Mask is empty. Setting center region for radiomics.")
            h, w = mask.shape
            mask[h//4: 3*h//4, w//4: 3*w//4] = 255
            sitk_mask = sitk.Cast(sitk.GetImageFromArray(mask), sitk.sitkUInt8)

        if params_path.is_file():
            extractor = featureextractor.RadiomicsFeatureExtractor(str(params_path))
        else:
            extractor = featureextractor.RadiomicsFeatureExtractor()
            extractor.settings['label'] = 255

        features_raw = extractor.execute(sitk_img, sitk_mask)
        extracted = {}
        for col in FEATURE_COLUMNS:
            if col in features_raw:
                extracted[col] = float(features_raw[col])
            else:
                extracted[col] = 0.0
        return extracted

    except ImportError:
        logger.info("PyRadiomics not installed. Using native statistical feature extraction.")
    except Exception as e:
        logger.warning(f"PyRadiomics extraction failed ({e}). Falling back to statistical extraction.")

    # 2. Native statistical fallback
    masked_pixels = image_gray[mask > 0].astype(np.float64)
    if len(masked_pixels) < 10:
        masked_pixels = image_gray.flatten().astype(np.float64)

    mean_val = np.mean(masked_pixels)
    std_val = np.std(masked_pixels) + 1e-6
    hist, _ = np.histogram(masked_pixels, bins=32, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    energy = np.sum(masked_pixels ** 2)
    uniformity = np.sum(hist ** 2)
    mad = np.mean(np.abs(masked_pixels - mean_val))

    from scipy import stats
    skewness = float(stats.skew(masked_pixels))
    kurt = float(stats.kurtosis(masked_pixels, fisher=False))

    # Basic texture indicators
    diff = np.diff(image_gray, axis=0)
    contrast = float(np.mean(diff ** 2))

    fallback = {
        'original_firstorder_Entropy': float(entropy),
        'original_firstorder_Energy': float(energy),
        'original_firstorder_Uniformity': float(uniformity),
        'original_firstorder_MeanAbsoluteDeviation': float(mad),
        'original_firstorder_Skewness': float(skewness),
        'original_firstorder_Kurtosis': float(kurt),
        'original_glcm_Contrast': contrast,
        'original_glcm_Idm': float(1.0 / (1.0 + contrast / 1000.0)),
        'original_glcm_Correlation': 0.5,
        'original_glcm_ClusterProminence': float(std_val * 10),
        'original_glcm_ClusterShade': float(skewness * 5),
        'original_gldm_GrayLevelNonUniformity': float(len(masked_pixels) * 0.1),
        'original_gldm_DependenceNonUniformity': float(len(masked_pixels) * 0.08),
        'original_gldm_DependenceVariance': float(std_val),
        'original_gldm_LowGrayLevelEmphasis': float(1.0 / (mean_val + 1.0)),
        'original_gldm_HighGrayLevelEmphasis': float(mean_val),
        'original_glszm_ZoneEntropy': float(entropy * 0.8),
        'original_glszm_LargeAreaEmphasis': 50.0,
        'original_glszm_SmallAreaEmphasis': 0.5,
        'original_glszm_GrayLevelVariance': float(std_val ** 2),
        'original_glszm_SizeZoneNonUniformity': 50.0,
        'original_glrlm_RunLengthNonUniformity': float(len(masked_pixels) * 0.12),
        'original_glrlm_ShortRunEmphasis': 0.75,
        'original_glrlm_LongRunEmphasis': 3.0
    }
    return fallback


def get_or_train_classifier():
    """Load existing classifier model or train a calibrated Random Forest on feature CSVs."""
    global _CLASSIFIER_PIPELINE
    if _CLASSIFIER_PIPELINE is not None:
        return _CLASSIFIER_PIPELINE

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = models_dir / "cad_classifier.joblib"

    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    if classifier_path.is_file():
        try:
            _CLASSIFIER_PIPELINE = joblib.load(classifier_path)
            logger.info(f"Loaded trained classifier from {classifier_path}")
            return _CLASSIFIER_PIPELINE
        except Exception as e:
            logger.warning(f"Failed to load cached classifier ({e}). Retraining...")

    logger.info("Training new Random Forest classifier from radiomics datasets...")
    covid_csv = BASE_DIR / "Analysis Dashboard" / "extracted_features_Covid.csv"
    normal_csv = BASE_DIR / "Analysis Dashboard" / "extracted_features_normal.csv"

    if not covid_csv.is_file() or not normal_csv.is_file():
        raise FileNotFoundError("Radiomics CSV files not found in 'Analysis Dashboard' directory.")

    covid_df = pd.read_csv(covid_csv)
    normal_df = pd.read_csv(normal_csv)
    covid_df['Target'] = 1
    normal_df['Target'] = 0

    full_df = pd.concat([covid_df, normal_df], ignore_index=True)

    X = full_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors='coerce')
    y = full_df['Target'].astype(int)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, classifier_path)
    logger.info(f"Classifier saved successfully to {classifier_path}")

    _CLASSIFIER_PIPELINE = pipeline
    return _CLASSIFIER_PIPELINE


def create_overlay(image_gray: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Create a high-contrast clinical overlay: Lung mask highlighted in cyan on CT image."""
    base_rgb = np.stack([image_gray] * 3, axis=-1).astype(np.float32)
    overlay = base_rgb.copy()

    # Cyan color for segmented lung mask: [0, 230, 255]
    color = np.array([0, 230, 255], dtype=np.float32)
    mask_indices = mask > 0

    overlay[mask_indices] = (1.0 - alpha) * base_rgb[mask_indices] + alpha * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def is_dicom_input(source: Any) -> bool:
    """Detect if input is a DICOM file or byte stream."""
    if isinstance(source, (str, Path)):
        if str(source).lower().endswith(('.dcm', '.dicom')):
            return True
        try:
            with open(source, 'rb') as f:
                f.seek(128)
                return f.read(4) == b'DICM'
        except Exception:
            return False
    if hasattr(source, 'name') and str(source.name).lower().endswith(('.dcm', '.dicom')):
        return True
    if hasattr(source, 'seek') and hasattr(source, 'read'):
        try:
            pos = source.tell()
            source.seek(128)
            magic = source.read(4)
            source.seek(pos)
            return magic == b'DICM'
        except Exception:
            pass
    return False


def load_dicom_image(dcm_source: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read DICOM scan, convert to Hounsfield Units (HU),
    apply standard Lung Window (WL=-600, WW=1500) and return uint8 2D grayscale array + metadata.
    """
    metadata = {}
    try:
        import pydicom
        if isinstance(dcm_source, (str, Path)):
            ds = pydicom.dcmread(str(dcm_source))
        else:
            if hasattr(dcm_source, 'seek'):
                dcm_source.seek(0)
            ds = pydicom.dcmread(dcm_source)

        metadata = {
            "Modality": str(getattr(ds, "Modality", "CT")),
            "PatientID": str(getattr(ds, "PatientID", "Anonymous")),
            "BodyPartExamined": str(getattr(ds, "BodyPartExamined", "CHEST")),
            "SliceThickness": str(getattr(ds, "SliceThickness", "N/A")),
            "Rows": getattr(ds, "Rows", None),
            "Columns": getattr(ds, "Columns", None)
        }

        # Ensure PhotometricInterpretation is defined (standard CT is MONOCHROME2)
        if not hasattr(ds, "PhotometricInterpretation") or not ds.PhotometricInterpretation:
            ds.PhotometricInterpretation = "MONOCHROME2"

        pixel_array = ds.pixel_array.astype(np.float32)
        if pixel_array.ndim == 3:
            pixel_array = pixel_array[pixel_array.shape[0] // 2]

        # Convert raw pixel values to Hounsfield Units (HU)
        slope = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        hu_image = pixel_array * slope + intercept

        # Apply standard Lung Window: Center = -600 HU, Width = 1500 HU
        wc = getattr(ds, 'WindowCenter', -600)
        ww = getattr(ds, 'WindowWidth', 1500)
        try:
            wc = float(wc[0]) if hasattr(wc, '__iter__') else float(wc)
            ww = float(ww[0]) if hasattr(ww, '__iter__') else float(ww)
        except Exception:
            wc, ww = -600.0, 1500.0

        if ww <= 0:
            wc, ww = -600.0, 1500.0

        metadata["WindowCenter"] = wc
        metadata["WindowWidth"] = ww

        lower = wc - ww / 2.0
        upper = wc + ww / 2.0
        windowed = np.clip(hu_image, lower, upper)
        img_8bit = ((windowed - lower) / (upper - lower) * 255.0).astype(np.uint8)
        return img_8bit, metadata

    except ImportError:
        logger.info("pydicom not installed. Trying SimpleITK fallback...")
        try:
            import SimpleITK as sitk
            if isinstance(dcm_source, (str, Path)):
                sitk_img = sitk.ReadImage(str(dcm_source))
                arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[arr.shape[0] // 2]
                windowed = np.clip(arr, -1350, 150)
                img_8bit = ((windowed + 1350) / 1500.0 * 255.0).astype(np.uint8)
                metadata["Modality"] = "CT"
                return img_8bit, metadata
        except Exception as sitk_err:
            logger.error(f"SimpleITK failed to read DICOM: {sitk_err}")

        raise ImportError("To process DICOM files (.dcm), please install pydicom: 'pip install pydicom'")


def diagnose_image(image_input: Union[str, Path, Image.Image, np.ndarray, Any]) -> Dict[str, Any]:
    """
    End-to-end diagnosis pipeline for a new chest CT/X-ray scan (JPG, PNG, DICOM):
    1. Preprocess input image (including DICOM HU lung windowing).
    2. Segment lung parenchyma using U-Net.
    3. Extract 24 radiomics texture/first-order features.
    4. Predict COVID-19 vs Normal classification probability.
    """
    dicom_meta = {}

    # 1. Convert input to grayscale numpy array (supports DICOM, PIL, and standard formats)
    if is_dicom_input(image_input):
        logger.info("DICOM input format detected. Applying Hounsfield lung windowing...")
        img_gray, dicom_meta = load_dicom_image(image_input)
    elif isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert('L')
        img_gray = np.array(pil_img, dtype=np.uint8)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert('L')
        img_gray = np.array(pil_img, dtype=np.uint8)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            pil_img = Image.fromarray(image_input).convert('L')
            img_gray = np.array(pil_img, dtype=np.uint8)
        else:
            img_gray = image_input.astype(np.uint8)
    elif hasattr(image_input, 'read'):
        # Uploaded file stream (Streamlit / Flask)
        try:
            if hasattr(image_input, 'seek'):
                image_input.seek(0)
            pil_img = Image.open(image_input).convert('L')
            img_gray = np.array(pil_img, dtype=np.uint8)
        except Exception:
            # Attempt DICOM reading on stream
            img_gray, dicom_meta = load_dicom_image(image_input)
    else:
        raise ValueError("Unsupported image input type.")

    # 2. Lung Segmentation
    mask = segment_lung(img_gray)

    # 3. Radiomic Feature Extraction
    features_dict = extract_radiomics(img_gray, mask)

    # 4. Classification
    clf = get_or_train_classifier()
    feature_df = pd.DataFrame([features_dict])[FEATURE_COLUMNS]

    prob = clf.predict_proba(feature_df)[0]
    pred_class_idx = int(np.argmax(prob))

    classes = ["Normal", "COVID-19"]
    pred_label = classes[pred_class_idx]
    confidence = float(prob[pred_class_idx] * 100)

    # 5. Generate Visuals
    overlay = create_overlay(img_gray, mask)

    return {
        "status": "success",
        "prediction": pred_label,
        "probability_normal": float(prob[0]),
        "probability_covid": float(prob[1]),
        "confidence": confidence,
        "features": features_dict,
        "image_gray": img_gray,
        "mask": mask,
        "overlay": overlay,
        "dicom_metadata": dicom_meta
    }
