# HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH HỆ THỐNG CAD-ANALYSIS-DASHBOARD

Hệ thống **CAD Analysis Dashboard** là nền tảng phân tích đặc trưng hình ảnh phóng xạ (Radiomics) và phân loại tổn thương phổi (COVID-19 vs Normal) dựa trên ảnh CT và mô hình học sâu U-Net kết hợp các thuật toán Machine Learning.

---

## 1. Yêu cầu hệ thống (System Requirements)

### 1.1. Phần cứng (Hardware)
- **CPU**: Tối thiểu 4 lõi (Khuyến nghị 8 lõi trở lên).
- **RAM**: Tối thiểu 8 GB (Khuyến nghị 16 GB trở lên khi chạy feature extraction hoặc huấn luyện U-Net).
- **Ổ cứng**: Còn trống tối thiểu 5 GB (Khoảng 20 GB nếu tải dataset ảnh CT).
- **GPU** (Tùy chọn): NVIDIA GPU hỗ trợ CUDA (nếu muốn huấn luyện lại U-Net cục bộ). Để chạy Dashboard, chỉ cần CPU thông thường.

### 1.2. Hệ điều hành & Phần mềm
- **Hệ điều hành**: Windows 10/11, Ubuntu 20.04/22.04 LTS, hoặc macOS.
- **Python**: **Python 3.10.x** (Khuyến nghị sử dụng Python 3.10 vì `pyradiomics` và `tensorflow 2.8 - 2.15` hỗ trợ ổn định nhất).
- **Công cụ dòng lệnh**: PowerShell / Command Prompt (Windows) hoặc Bash / Zsh (Linux/macOS).

---

## 2. Các gói Requirement (Thư viện phụ thuộc)

Dự án được chia thành 2 mức độ cài đặt:
1. **Mức 1 - Chạy Dashboard & EDA (`requirements.txt`)**: Dành cho người dùng muốn khởi chạy web dashboard tương tác, khám phá dữ liệu và so sánh mô hình ML trên dữ liệu radiomics đã trích xuất.
2. **Mức 2 - Toàn diện ML & Trích xuất Radiomics (`requirements-ml.txt`)**: Dành cho nhà nghiên cứu muốn chạy lại các file Jupyter Notebook trong thư mục `ML/` (huấn luyện U-Net, cắt mask, trích xuất đặc trưng góc bằng PyRadiomics).

---

## 3. Hướng dẫn cài đặt từng bước

### Bước 1: Chuẩn bị môi trường Python (Python 3.10)

#### Cách 1: Sử dụng `venv` (Mặc định của Python)
- **Trên Windows (PowerShell/CMD):**
  ```powershell
  # Di chuyển vào thư mục dự án
  cd CAD-Analysis-Dashboard

  # Tạo virtual environment
  python -m venv venv

  # Kích hoạt virtual environment
  venv\Scripts\activate
  ```
- **Trên Linux/macOS:**
  ```bash
  cd CAD-Analysis-Dashboard
  python3 -m venv venv
  source venv/bin/activate
  ```

#### Cách 2: Sử dụng Anaconda / Miniconda (Khuyên dùng nếu chạy ML/PyRadiomics trên Windows)
```bash
conda create -n cad_env python=3.10 -y
conda activate cad_env
```

---

### Bước 2: Nâng cấp pip và cài đặt thư viện

1. **Nâng cấp pip:**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

2. **Cài đặt thư viện chạy Dashboard & Phân tích:**
   ```bash
   pip install -r requirements.txt
   ```

3. *(Tùy chọn)* **Cài đặt thư viện huấn luyện Deep Learning & PyRadiomics:**
   > **Lưu ý trên Windows:** `pyradiomics` yêu cầu Microsoft C++ Build Tools nếu cài từ source. Nên ưu tiên cài trên Python 3.10 để nhận wheel có sẵn.
   ```bash
   pip install -r requirements-ml.txt
   ```

---

### Bước 3: Khởi tạo Database SQLite (Nếu cần reset hoặc build lại)

Dữ liệu đặc trưng Radiomics trích xuất sẵn từ ảnh COVID và Normal nằm trong 2 file CSV:
- `Analysis Dashboard/extracted_features_Covid.csv`
- `Analysis Dashboard/extracted_features_normal.csv`

File cơ sở dữ liệu `radiomics_data.db` đã được khởi tạo sẵn. Nếu muốn tạo lại database từ 2 file CSV:
```bash
# Chạy script setup database từ thư mục gốc của project:
python "Analysis Dashboard/database_setup.py"
```

---

### Bước 4: Khởi chạy Streamlit Dashboard

Để mở ứng dụng giao diện web Dashboard tương tác:
```bash
streamlit run "Analysis Dashboard/dashboard.py"
```

Sau khi chạy lệnh trên, ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ:
```
Local URL: http://localhost:8501
Network URL: http://<địa_chỉ_IP_mạng>:8501
```

---

## 4. Các tính năng chính trên Web Dashboard

1. **Sidebar Điều hướng & Chuẩn hóa:**
   - Chọn trang phân tích: **Textual Analysis** (Phân tích thống kê & Báo cáo) hoặc **Visual Comparisons** (Biểu đồ tương tác).
   - Chọn phương pháp chuẩn hóa dữ liệu: **Min-Max**, **Z-score**, hoặc **Max-Abs**.
2. **Trang Textual Analysis:**
   - Lọc dữ liệu theo ca bệnh: **Both** (Tất cả), **COVID** (Ca bệnh dương tính), **Normal** (Ca bình thường).
   - Tổng quan dữ liệu, đếm Missing values và xử lý trực tiếp (Fill 0, Fill Mean/Median, Drop rows).
   - Thẻ thống kê mô tả (Mean, Median, Mode, Std Dev, IQR).
   - Ma trận tương quan (Correlation Matrix) và Top 5 đặc trưng quan trọng nhất liên kết với nhãn bệnh.
   - Báo cáo loại bỏ ngoại lai (Outlier Removal Summary).
   - Kiểm định phân phối chuẩn (Anderson-Darling Test & Kolmogorov-Smirnov Test).
   - Kiểm định giả thuyết t-Test 2 mẫu độc lập (Two-sample t-test) so sánh nhóm COVID vs Normal.
   - Phân tích giảm chiều PCA (Explained Variance Ratio).
   - Huấn luyện & Đánh giá so sánh 3 mô hình phân loại: **SVM**, **Random Forest**, **Logistic Regression** (với 5-Fold Cross Validation).
3. **Trang Visual Comparisons:**
   - Heatmap ma trận tương quan đa chiều (Plotly).
   - Biểu đồ thanh Top correlated features.
   - Biểu đồ phân phối Histogram, Box Plot và Scatter Plot đối chiếu với Target.
   - Biểu đồ QQ Plot và Histogram trước và sau chuẩn hóa.
   - Biểu đồ chiếu PCA 2D Scatter Plot.
   - Đồ thị đường cong ROC-AUC, ma trận nhầm lẫn (Confusion Matrix) và Feature Importance của từng mô hình ML.

---

## 5. Cấu trúc thư mục dự án

```text
CAD-Analysis-Dashboard/
├── Analysis Dashboard/
│   ├── dashboard.py                  # Mã nguồn chính của Streamlit Dashboard
│   ├── database_setup.py             # Script nạp CSV vào SQLite DB (đã fix path động)
│   ├── ex.py                         # Script kiểm tra truy vấn nhanh DB
│   ├── extracted_features_Covid.csv  # 2.5MB đặc trưng Radiomics trích xuất ca COVID
│   ├── extracted_features_normal.csv # 1.3MB đặc trưng Radiomics trích xuất ca Normal
│   ├── params.yaml                   # Cấu hình tham số trích xuất PyRadiomics
│   └── radiomics_data.db             # Cơ sở dữ liệu SQLite chứa bảng radiomic_features
├── ML/
│   ├── Feature Extraction/
│   │   ├── Radiomics_Anglewise_Covid.ipynb   # Trích xuất góc (0, 45, 90, 135 độ) ảnh COVID
│   │   └── Radiomics_Anglewise_Normal.ipynb  # Trích xuất góc ảnh Normal
│   ├── Random_Classifier.ipynb               # Huấn luyện Random Forest, SMOTE, KNNImputer
│   └── UNET Training/
│       ├── Data Split.ipynb                  # Chia tập Train/Val/Test cho ảnh và mask
│       ├── Unet Training.ipynb               # Định nghĩa kiến trúc U-Net và huấn luyện
│       ├── Unet_new.ipynb                    # Pipeline suy luận và tạo mask tự động
│       ├── lung_segmentation_unet .h5        # Trọng số mô hình U-Net đã train (~372MB)
│       └── predictions.npy                   # Kết quả dự đoán mask lưu dạng numpy
├── modules/
│   └── eda.py                        # Thư viện hàm xử lý EDA, thống kê, PCA, ML
├── requirements.txt                  # Danh sách thư viện cho Dashboard & EDA
├── requirements-ml.txt               # Danh sách thư viện cho Deep Learning & Radiomics
├── SETUP_GUIDE.md                    # Tài liệu hướng dẫn cài đặt & vận hành (file này)
└── README.md                         # Giới thiệu tổng quan dự án
```

---

## 6. Xử lý sự cố thường gặp (Troubleshooting)

| Vấn đề / Lỗi | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'modules'` | Chạy streamlit từ bên trong thư mục con khiến Python không tìm thấy `modules` ở root. | Đã được xử lý tự động trong `dashboard.py` qua `sys.path`. Hãy chạy lệnh: `streamlit run "Analysis Dashboard/dashboard.py"` từ thư mục gốc. |
| `sqlite3.OperationalError: no such table: radiomic_features` | Database chưa được tạo hoặc đường dẫn database sai thư mục làm việc. | Chạy lệnh `python "Analysis Dashboard/database_setup.py"` để khởi tạo lại database. |
| `FileNotFoundError: ... extracted_features_Covid.csv` | Đường dẫn file bị hardcode trên máy phát triển cũ. | Đã được cập nhật sang đường dẫn tương đối động (`Path(__file__).resolve().parent`). |
| `error: Microsoft Visual C++ 14.0 or greater is required` khi cài `pyradiomics` | Thiếu trình biên dịch C++ trên Windows khi build từ source. | Cài đặt **Visual Studio C++ Build Tools** hoặc dùng conda: `conda install -c conda-forge pyradiomics`. Nếu chỉ chạy Dashboard, không cần cài pyradiomics. |
| `ModuleNotFoundError: No module named 'numpy'` khi build `pyradiomics` | PyPI package `pyradiomics-3.0.1` thiếu khai báo build dependency trong `pyproject.toml` nên pip build isolation không có numpy. | 1. Nếu chỉ chạy Dashboard: chỉ cần chạy `pip install -r requirements.txt` (không cần pyradiomics).<br>2. Nếu cần pyradiomics: chạy `pip install wheel numpy` trước, sau đó chạy `pip install pyradiomics --no-build-isolation` (hoặc dùng conda). |
| Port 8501 đã bị chiếm dụng | Có một phiên bản Streamlit khác đang chạy ngầm. | Chạy lệnh chỉ định cổng khác: `streamlit run "Analysis Dashboard/dashboard.py" --server.port 8502`. |

---

## 7. Hướng dẫn chạy chẩn đoán với ảnh mới (Inference on New Images)

Dự án đã được tích hợp đầy đủ pipeline chẩn đoán tự động cho ảnh mới qua 2 cách:

### Cách 1: Chẩn đoán trực tiếp trên Web Dashboard (Giao diện trực quan)
1. Khởi chạy Dashboard:
   ```bash
   streamlit run "Analysis Dashboard/dashboard.py"
   ```
2. Trên thanh menu bên trái, chọn mục: **`New Diagnosis (Chẩn đoán ảnh mới)`**.
3. Kéo thả hoặc tải lên ảnh chụp CT lồng ngực (định dạng `.jpg`, `.png`, `.jpeg`).
4. Nhấn nút **🚀 Run Clinical Diagnosis**.
5. Hệ thống sẽ tự động hiển thị:
   - Kết quả phân loại: **COVID-19** hay **Normal** kèm thanh đo phần trăm xác suất rủi ro.
   - Ảnh gốc CT, Mặt nạ phân vùng phổi tạo bởi **U-Net** và Ảnh phủ màu tổn thương (Cyan Overlay).
   - Bảng 24 đặc trưng Radiomics trích xuất trực tiếp từ ảnh.

### Cách 2: Chạy chẩn đoán qua dòng lệnh (CLI Script - `predict.py`)
Phù hợp khi cần tích hợp vào backend hoặc chạy chẩn đoán hàng loạt thư mục ảnh:

- **Chẩn đoán 1 ảnh:**
  ```bash
  python predict.py --image "duong_dan/toi/anh_ct.jpg" --output "ket_qua_chan_doan"
  ```
- **Chẩn đoán hàng loạt cả thư mục ảnh:**
  ```bash
  python predict.py --dir "thu_muc_chua_anh_ct/" --output "ket_qua_chan_doan"
  ```
  Kết quả mặt nạ phổi (`_mask.png`), ảnh phủ màu (`_overlay.png`) và file tổng hợp kết quả (`batch_diagnosis_summary.csv`) sẽ được tự động lưu vào thư mục đầu ra.
