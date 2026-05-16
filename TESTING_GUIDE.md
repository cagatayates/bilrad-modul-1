# Radar Deinterleaving U-Net - Test Kılavuzu

Bu kılavuz, eğitilmiş radar deinterleaving modelini test etmek için gerekli adımları açıklar.

## 📁 Dosya Yapısı

```
├── train.py                    # Eğitim scripti (validation verisini kaydeder)
├── test_model.py              # Validation verisi üzerinde test (tek window)
├── test_simple_full_frame.py  # Validation verisi üzerinde test (çoklu window)
├── inference.py               # Tek frame inference
├── modules.py                 # U-Net modeli
├── config.json                # Konfigürasyon
└── checkpoints/               # Eğitilmiş modeller
    ├── unet1d_pit_case1_N3_best.pth
    ├── unet1d_pit_case1_N3_last.pth
    └── val_data.npz           # Validation verisi (eğitim sırasında oluşur)
```

## 🚀 Kullanım Adımları

### 1. Model Eğitimi (Validation verisini kaydetmek için)

```bash
python train.py --config config.json
```

Bu komut:
- Modeli eğitir
- Validation verisini `checkpoints/val_data.npz` olarak kaydeder
- En iyi modeli `checkpoints/unet1d_pit_case1_N3_best.pth` olarak kaydeder

### 2. Validation Verisi Üzerinde Test

#### A. Tek Window Test (`test_model.py`)

```bash
# Temel test (5 frame görselleştir)
python test_model.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --val_data checkpoints/val_data.npz

# Daha fazla frame test et
python test_model.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --val_data checkpoints/val_data.npz --frames 10

# Belirli bir frame'den başla
python test_model.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --val_data checkpoints/val_data.npz --start_frame 5 --frames 3

# Plotları kaydet
python test_model.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --val_data checkpoints/val_data.npz --save_plots --output_dir test_results
```

#### B. Çoklu Window Test (`test_simple_full_frame.py`)

```bash
# Belirli bir frame'in tüm window'larını test et
python test_simple_full_frame.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --val_data checkpoints/val_data.npz --frame_id 0

# Belirli window'ları test et
python test_simple_full_frame.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --val_data checkpoints/val_data.npz --start_window 0 --windows 5

# Plotları kaydet
python test_simple_full_frame.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --val_data checkpoints/val_data.npz --frame_id 0 --save_plots --output_dir test_results_full_frame
```

### 3. Tek Frame Inference

```bash
# Temel inference (ground truth olmadan)
python inference.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --data data/deinterleaving_u_net_case_2_v0_scenario_data.npy --frame_idx 0

# Ground truth ile PIT assignment
python inference.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --data data/deinterleaving_u_net_case_2_v0_scenario_data.npy --labels data/deinterleaving_u_net_case_2_v0_scenario_labels.npy --frame_idx 0

# Plot kaydet
python inference.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --data data/deinterleaving_u_net_case_2_v0_scenario_data.npy --frame_idx 0 --save_plot frame_0_result.png
```

## 📊 Görselleştirme Açıklaması

### Test Model Görselleştirmesi (`test_model.py`)

Her window için şu grafikler oluşturulur:

1. **Magnitude + I/Q Radar Sinyalleri**: Ham radar verisi (magnitude + I/Q kanalları)
2. **Ground Truth vs Prediction**: Her verici için karşılaştırma
   - Renkli çizgi: Ground Truth
   - Kırmızı scatter: Model tahmini (nokta)
   - Kırmızı kesikli çizgi: Model tahmini (çizgi)

### Test Simple Full Frame Görselleştirmesi (`test_simple_full_frame.py`)

Çoklu window'ları birleştirerek şu grafikler oluşturulur:

1. **Magnitude Radar Sinyali**: Birleştirilmiş window'lar
2. **Ground Truth vs Prediction**: Her verici için karşılaştırma
   - Renkli çizgi: Ground Truth
   - Kırmızı scatter: Model tahmini (nokta)
   - Kırmızı kesikli çizgi: Model tahmini (çizgi)
   - Kırmızı dikey çizgiler: Window sınırları
3. **Performans Metrikleri**: Sol alt köşede
   - Dice Score, F1 Score, Precision, Recall
   - Ortalama aktif verici sayısı

### Inference Görselleştirmesi (`inference.py`)

1. **I/Q Radar Sinyalleri**: Ham radar verisi
2. **Verici Tahminleri**: Her verici için
   - Düz çizgi: Olasılık skoru (0-1)
   - Kesikli çizgi: Binary maske
   - Kırmızı noktalı: Ground Truth (varsa)

## 🔧 Parametreler

### Test Model Parametreleri (`test_model.py`)

- `--checkpoint`: Model checkpoint dosyası yolu
- `--val_data`: Validation verisi dosyası yolu
- `--frames`: Test edilecek frame sayısı (varsayılan: 5)
- `--start_frame`: Başlangıç frame indeksi (varsayılan: 0)
- `--device`: Kullanılacak cihaz (cpu/cuda/auto)
- `--save_plots`: Plotları dosyaya kaydet
- `--output_dir`: Çıktı dizini (varsayılan: test_results)

### Test Simple Full Frame Parametreleri (`test_simple_full_frame.py`)

- `--checkpoint`: Model checkpoint dosyası yolu
- `--val_data`: Validation verisi dosyası yolu
- `--frame_id`: Test edilecek frame ID'si (frame_id belirtilirse, o frame'in tüm window'ları kullanılır)
- `--start_window`: Başlangıç window indeksi (frame_id belirtilmemişse)
- `--windows`: Test edilecek window sayısı (frame_id belirtilmemişse, varsayılan: 5)
- `--device`: Kullanılacak cihaz (cpu/cuda/auto)
- `--save_plots`: Plotları dosyaya kaydet
- `--output_dir`: Çıktı dizini (varsayılan: test_results_consecutive)

### Inference Parametreleri

- `--checkpoint`: Model checkpoint dosyası yolu
- `--data`: Test verisi dosyası yolu
- `--labels`: Ground truth etiketleri (opsiyonel)
- `--frame_idx`: Test edilecek frame indeksi
- `--threshold`: Binary maske eşiği (varsayılan: 0.5)
- `--save_plot`: Plot kaydetme yolu

## 📈 Metrikler

### Test Model Metrikleri

- **Dice Score**: Her verici için overlap kalitesi
- **F1 Score**: Genel sınıflandırma performansı
- **PIT Assignment**: Optimal eşleştirme bilgisi

### Inference Metrikleri

- **Probability Scores**: Her zaman noktası için olasılık
- **Binary Masks**: Eşik sonrası binary maskeler
- **PIT Assignment**: Ground truth varsa optimal eşleştirme

## 🎯 Örnek Kullanım Senaryoları

### Senaryo 1: Hızlı Test
```bash
# Eğitim sonrası hızlı kontrol (tek window)
python test_model.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --val_data checkpoints/val_data.npz --frames 3

# Eğitim sonrası hızlı kontrol (çoklu window)
python test_simple_full_frame.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --val_data checkpoints/val_data.npz --frame_id 0
```

### Senaryo 2: Detaylı Analiz
```bash
# Tüm validation verisini test et ve kaydet (tek window)
python test_model.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --val_data checkpoints/val_data.npz --frames 20 --save_plots --output_dir detailed_analysis

# Belirli frame'lerin tam analizi (çoklu window)
python test_simple_full_frame.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --val_data checkpoints/val_data.npz --frame_id 0 --save_plots --output_dir detailed_analysis_full_frame
```

### Senaryo 3: Tek Frame İnceleme
```bash
# Belirli bir frame'i detaylı incele (inference)
python inference.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --data data/deinterleaving_u_net_case_2_v0_scenario_data.npy --labels data/deinterleaving_u_net_case_2_v0_scenario_labels.npy --frame_idx 15 --save_plot frame_15_analysis.png

# Belirli bir frame'in tüm window'larını incele
python test_simple_full_frame.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --val_data checkpoints/val_data.npz --frame_id 15 --save_plots --output_dir frame_15_analysis
```

### Senaryo 4: Performans Karşılaştırması
```bash
# Farklı window sayıları ile test
python test_simple_full_frame.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --val_data checkpoints/val_data.npz --start_window 0 --windows 3 --save_plots --output_dir test_3_windows
python test_simple_full_frame.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --val_data checkpoints/val_data.npz --start_window 0 --windows 5 --save_plots --output_dir test_5_windows
python test_simple_full_frame.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --val_data checkpoints/val_data.npz --start_window 0 --windows 10 --save_plots --output_dir test_10_windows
```

### Senaryo 5: Farklı Eşikler Test Etme
```bash
# Farklı eşik değerleri ile test (inference)
for threshold in 0.3 0.5 0.7; do
    python inference.py --checkpoint checkpoints/unet1d_pit_case1_N3_best.pth --data data/deinterleaving_u_net_case_2_v0_scenario_data.npy --frame_idx 0 --threshold $threshold --save_plot "frame_0_threshold_${threshold}.png"
done
```

## 🐛 Sorun Giderme

### Yaygın Hatalar

1. **CUDA Out of Memory**: `--device cpu` kullanın
2. **Model Bulunamadı**: Checkpoint dosya yolunu kontrol edin
3. **Validation Data Bulunamadı**: Önce `train.py` çalıştırın
4. **Plot Görünmüyor**: `--save_plots` veya `--save_plot` kullanın

### Debug İpuçları

- Model yüklenirken epoch ve loss bilgilerini kontrol edin
- Validation verisi yüklenirken sample sayısını kontrol edin
- Plot kaydedilirken dosya yollarını kontrol edin

## 📝 Notlar

- Validation verisi sadece eğitim sırasında oluşturulur
- Model checkpoint'leri otomatik olarak kaydedilir
- Plotlar yüksek çözünürlükte (300 DPI) kaydedilir
- PIT assignment sadece ground truth varsa uygulanır
- `test_model.py`: Tek window'ları test eder, hızlı analiz için ideal
- `test_simple_full_frame.py`: Çoklu window'ları birleştirir, tam frame analizi için ideal
- `inference.py`: Ham veri üzerinde inference yapar, gerçek zamanlı kullanım için ideal
- Performans metrikleri sadece `test_simple_full_frame.py`'de gösterilir
