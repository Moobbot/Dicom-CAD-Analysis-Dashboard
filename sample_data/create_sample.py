import numpy as np
from PIL import Image
from pathlib import Path

# Create a sample synthetic CT lung slice
h, w = 512, 512
y, x = np.ogrid[:h, :w]
center_y, center_x = h // 2, w // 2

# Background body contour
body_mask = ((x - center_x) ** 2 / (0.42 * w) ** 2 + (y - center_y) ** 2 / (0.45 * h) ** 2) <= 1.0
img = np.zeros((h, w), dtype=np.uint8)
img[body_mask] = 140  # Soft tissue

# Ribs / spine (brighter bone structure)
spine_mask = ((x - center_x) ** 2 / (0.05 * w) ** 2 + (y - center_y - 0.3 * h) ** 2 / (0.06 * h) ** 2) <= 1.0
img[spine_mask] = 230

# Left and Right Lung cavities (darker regions)
left_lung = ((x - (center_x - 0.18 * w)) ** 2 / (0.16 * w) ** 2 + (y - (center_y - 0.05 * h)) ** 2 / (0.28 * h) ** 2) <= 1.0
right_lung = ((x - (center_x + 0.18 * w)) ** 2 / (0.16 * w) ** 2 + (y - (center_y - 0.05 * h)) ** 2 / (0.28 * h) ** 2) <= 1.0

# Add noise and texture inside lungs
noise = np.random.normal(35, 12, (h, w)).clip(10, 80).astype(np.uint8)
img[left_lung] = noise[left_lung]
img[right_lung] = noise[right_lung]

# Simulated lesion / ground glass opacity in peripheral right lung
lesion_mask = ((x - (center_x + 0.22 * w)) ** 2 / (0.07 * w) ** 2 + (y - (center_y + 0.08 * h)) ** 2 / (0.08 * h) ** 2) <= 1.0
img[lesion_mask & right_lung] = 110

sample_dir = Path("sample_data")
sample_dir.mkdir(exist_ok=True)
sample_path = sample_dir / "sample_chest_ct.jpg"
Image.fromarray(img).save(sample_path)
print(f"Demo image saved to: {sample_path}")

# Create synthetic DICOM file (.dcm) if pydicom is available
try:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dcm_path = sample_dir / "sample_chest_ct.dcm"
    dcm = FileDataset(str(dcm_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dcm.PatientName = "CAD^Patient^01"
    dcm.PatientID = "COVID-TEST-001"
    dcm.Modality = "CT"
    dcm.BodyPartExamined = "CHEST"
    dcm.SliceThickness = "2.5"
    dcm.Rows, dcm.Columns = h, w
    dcm.BitsAllocated = 16
    dcm.BitsStored = 16
    dcm.HighBit = 15
    dcm.PixelRepresentation = 1
    dcm.RescaleIntercept = -1024.0
    dcm.RescaleSlope = 1.0
    dcm.WindowCenter = -600.0
    dcm.WindowWidth = 1500.0

    # Simulated HU values (-1000 HU air to +300 HU tissue)
    hu = (img.astype(np.float32) / 255.0 * 1300.0 - 1000.0).astype(np.int16)
    stored_val = (hu - int(dcm.RescaleIntercept)).astype(np.uint16)
    dcm.PixelData = stored_val.tobytes()
    dcm.save_as(str(dcm_path))
    print(f"Demo DICOM CT scan saved to: {dcm_path}")
except ImportError:
    pass
