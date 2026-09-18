import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Set up figure with high resolution
fig, ax = plt.subplots(figsize=(19, 8), dpi=300)
fig.patch.set_facecolor('#0b1329')  # Deep dark navy background
ax.set_facecolor('#0b1329')

# Define stages
stages = [
    {
        "step": "ĐẦU VÀO",
        "title": "Ảnh Chụp Phổi",
        "sub": "Chest CT / X-ray Scan",
        "color": "#38bdf8",  # Light blue
        "border": "#0284c7",
        "items": [
            "• Định dạng: JPG, PNG, DICOM",
            "• Kênh màu: Grayscale",
            "• Kích thước: Đa độ phân giải",
            "• Đối tượng: Lát cắt lồng ngực"
        ]
    },
    {
        "step": "BƯỚC 1",
        "title": "Phân Vùng Phổi U-Net",
        "sub": "Deep Learning Segmentation",
        "color": "#06b6d4",  # Cyan
        "border": "#0891b2",
        "items": [
            "• Kiến trúc: 2D U-Net CNN",
            "• Input: Resize 128x128, norm [0, 1]",
            "• Trọng số: lung_segmentation_unet",
            "• Output: Mask nhị phân nhu mô phổi"
        ]
    },
    {
        "step": "BƯỚC 2",
        "title": "Trích Xuất Radiomics",
        "sub": "24 Biomarkers Extraction",
        "color": "#818cf8",  # Indigo
        "border": "#4f46e5",
        "items": [
            "• Engine: PyRadiomics (params.yaml)",
            "• First-Order (6): Entropy, Energy...",
            "• GLCM (5): Contrast, Correlation...",
            "• GLDM, GLSZM, GLRLM (13 biomarkers)"
        ]
    },
    {
        "step": "BƯỚC 3",
        "title": "Phân Loại Học Máy",
        "sub": "Machine Learning Classifier",
        "color": "#f59e0b",  # Amber/Orange
        "border": "#d97706",
        "items": [
            "• Model: Random Forest (200 Trees)",
            "• Pipeline: Imputer + StandardScaler",
            "• Dataset: 3.449 mẫu (COVID vs Normal)",
            "• Output: Xác suất p(COVID) vs p(Normal)"
        ]
    },
    {
        "step": "BƯỚC 4",
        "title": "Báo Cáo & Trực Quan",
        "sub": "Clinical Report & Delivery",
        "color": "#10b981",  # Emerald Green
        "border": "#059669",
        "items": [
            "• Chẩn đoán: COVID-19 vs Normal",
            "• Độ tin cậy: Confidence Score (%)",
            "• Visual: CT gốc vs Mask vs Cyan Overlay",
            "• Nền tảng: Web Dashboard & CLI predict"
        ]
    }
]

box_width = 3.2
box_height = 5.0
y_start = 1.3
x_spacing = 0.55
total_width = len(stages) * box_width + (len(stages) - 1) * x_spacing
start_x = 0.6

# Draw title and header
ax.text(9.5, 7.3, "QUY TRÌNH CHẨN ĐOÁN HÌNH ẢNH Y TẾ CAD-ANALYSIS PIPELINE",
        fontsize=22, fontweight='bold', color='#ffffff', ha='center', va='center')
ax.text(9.5, 6.75, "Tích hợp Học sâu (U-Net) + Trích xuất Đặc trưng Radiomics (24 Chỉ số) + Học máy (Random Forest)",
        fontsize=13, color='#94a3b8', ha='center', va='center')

for i, stage in enumerate(stages):
    x = start_x + i * (box_width + x_spacing)
    
    # Outer card shadow effect
    shadow = patches.FancyBboxPatch((x + 0.05, y_start - 0.05), box_width, box_height,
                                    boxstyle="round,pad=0.1,rounding_size=0.25",
                                    facecolor='#050b18', edgecolor='none', alpha=0.6, zorder=1)
    ax.add_patch(shadow)

    # Main Card Box
    card = patches.FancyBboxPatch((x, y_start), box_width, box_height,
                                  boxstyle="round,pad=0.1,rounding_size=0.25",
                                  facecolor='#1e293b', edgecolor=stage["border"],
                                  linewidth=2.2, zorder=2)
    ax.add_patch(card)

    # Step Badge Header Pill
    badge = patches.FancyBboxPatch((x + 0.3, y_start + box_height - 0.55), box_width - 0.6, 0.45,
                                   boxstyle="round,pad=0.06,rounding_size=0.15",
                                   facecolor=stage["color"], edgecolor='none', zorder=3)
    ax.add_patch(badge)
    ax.text(x + box_width / 2, y_start + box_height - 0.32, stage["step"],
            fontsize=12, fontweight='bold', color='#0f172a', ha='center', va='center', zorder=4)

    # Title & Subtitle
    ax.text(x + box_width / 2, y_start + box_height - 1.0, stage["title"],
            fontsize=14, fontweight='bold', color='#f8fafc', ha='center', va='center', zorder=4)
    ax.text(x + box_width / 2, y_start + box_height - 1.35, stage["sub"],
            fontsize=10.5, fontstyle='italic', color=stage["color"], ha='center', va='center', zorder=4)

    # Separator Line
    ax.plot([x + 0.3, x + box_width - 0.3], [y_start + box_height - 1.6, y_start + box_height - 1.6],
            color='#334155', linewidth=1.2, zorder=4)

    # Bullet Items
    for idx, item in enumerate(stage["items"]):
        ax.text(x + 0.25, y_start + box_height - 2.1 - idx * 0.72, item,
                fontsize=10.5, color='#cbd5e1', ha='left', va='center', zorder=4)

    # Connecting Arrows between cards
    if i < len(stages) - 1:
        arrow_x = x + box_width + 0.08
        arrow_y = y_start + box_height / 2
        ax.annotate('', xy=(arrow_x + x_spacing - 0.16, arrow_y), xytext=(arrow_x, arrow_y),
                    arrowprops=dict(arrowstyle="-|>", color='#38bdf8', lw=3.0,
                                    mutation_scale=22), zorder=5)

# Limits and cleanup
ax.set_xlim(0, 19)
ax.set_ylim(0.5, 8.0)
ax.axis('off')

# Save outputs
output_png = Path("cad_pipeline_diagram_vi.png")
plt.tight_layout()
plt.savefig(output_png, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight', dpi=300)
plt.close(fig)
print(f"High-res Vietnamese diagram saved to {output_png.resolve()}")
