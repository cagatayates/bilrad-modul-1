# pri_infer.py
import argparse, json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pri_net import PRINet
import os
import glob
from pri_features import extract_pri_features

def extract_features_from_iq_data(iq_data, fs_hz=3_000_000):
    """
    Ham I/Q verisinden öznitelik çıkarır (pri_features.py kullanarak).
    
    Args:
        iq_data: (N, 2) numpy array - I ve Q verileri
        fs_hz: float - örnekleme frekansı (Hz)
        
    Returns:
        features_dict: dict - tüm çıkarılan öznitelikler (histogram, ACF, feat_vector, vs.)
    """
    # I/Q verisini (N, 2) formatından (2, N) formatına çevir
    if iq_data.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) shape, got {iq_data.shape}")
    
    iq_transposed = iq_data.T  # (2, N) formatına çevir
    
    # pri_features.py kullanarak öznitelik çıkar
    features_dict = extract_pri_features(
        iq=iq_transposed,
        fs_hz=fs_hz,
        smooth_len=9,
        k_hi=4.0,
        k_lo=2.0,
        min_len_us=5.0,
        min_gap_us=2.0,
        nbins=512,
        tmax_us=15000.0,
        include_acf=True,
        acf_nlags=512,
        pri_window_pulses=8,
        pri_hop_pulses=4,
        pri_track_outlen=64
    )
    
    return features_dict

def visualize_raw_data_with_predictions(iq_data, predictions, save_path=None):
    """
    Ham I/Q verisini ve model tahminlerini görselleştirir.
    
    Args:
        iq_data: (N, 2) numpy array - I ve Q verileri
        predictions: dict - model tahminleri
        save_path: str - kaydetme yolu
    """
    I = iq_data[:, 0]
    Q = iq_data[:, 1]
    magnitude = np.sqrt(I**2 + Q**2)
    
    # Zaman ekseni
    time_axis = np.arange(len(I))
    
    # Grafik oluştur - tek grafikte I, Q, magnitude üst üste
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Ana grafik - I, Q, magnitude üst üste
    ax_main = axes[0]
    
    # I verisi
    ax_main.plot(time_axis, I, 'b-', linewidth=0.5, alpha=0.7, label='I (In-phase)')
    
    # Q verisi
    ax_main.plot(time_axis, Q, 'r-', linewidth=0.5, alpha=0.7, label='Q (Quadrature)')
    
    # Magnitude
    ax_main.plot(time_axis, magnitude, 'g-', linewidth=0.5, alpha=0.7, label='Magnitude (|I + jQ|)')
    
    ax_main.set_title('I/Q Verisi ve Magnitude - Üst Üste Görünüm', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Değer')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='upper right')
    
    # Model tahminleri
    axes[1].axis('off')
    
    # Tahmin bilgilerini göster
    pred_text = f"Model Tahminleri:\n"
    pred_text += f"PRI Modu: {predictions['pred_name'][0]}\n"
    if 'pri_us' in predictions:
        pri_value = predictions['pri_us'][0] if hasattr(predictions['pri_us'], '__len__') else predictions['pri_us']
        pred_text += f"Tahmin Edilen PRI: {pri_value:.2f} μs\n"
    if 'probs' in predictions:
        probs_array = predictions['probs'][0] if hasattr(predictions['probs'][0], '__len__') else predictions['probs']
        pred_text += f"Güven Skoru: {np.max(probs_array):.3f}\n"
    
    # Tüm sınıfların olasılıklarını göster
    if 'probs' in predictions and 'classes' in predictions:
        probs_array = predictions['probs'][0] if hasattr(predictions['probs'][0], '__len__') else predictions['probs']
        pred_text += f"\nTüm Sınıf Olasılıkları:\n"
        for i, prob in enumerate(probs_array):
            class_name = predictions['classes'][i] if 'classes' in predictions else f"Class {i}"
            pred_text += f"  {class_name}: {prob:.3f}\n"
    
    axes[1].text(0.1, 0.5, pred_text, transform=axes[1].transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Ham veri görselleştirmesi kaydedildi: {save_path}")
    
    plt.show()

def visualize_histogram(hist, bin_edges_us, predictions, save_path=None):
    """
    IPI histogramını görselleştirir.
    
    Args:
        hist: (512,) numpy array - histogram değerleri
        bin_edges_us: (513,) numpy array - bin edge'leri (μs cinsinden)
        predictions: dict - model tahminleri
        save_path: str - kaydetme yolu
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Histogram
    bin_centers = (bin_edges_us[:-1] + bin_edges_us[1:]) / 2
    
    ax1.bar(bin_centers, hist, width=(bin_edges_us[1] - bin_edges_us[0]) * 0.8, 
            alpha=0.7, color='skyblue', edgecolor='navy', linewidth=0.5)
    ax1.set_xlabel('IPI (Inter-Pulse Interval) [μs]')
    ax1.set_ylabel('Normalized Frequency')
    ax1.set_title('IPI Histogram (Inter-Pulse Interval Distribution)')
    ax1.grid(True, alpha=0.3)
    
    # Tahmin edilen PRI değerini histogram üzerinde göster
    if 'pri_us' in predictions:
        pri_value = predictions['pri_us'][0] if hasattr(predictions['pri_us'], '__len__') else predictions['pri_us']
        ax1.axvline(x=pri_value, color='red', linestyle='--', linewidth=2, 
                   label=f'Predicted PRI: {pri_value:.2f}μs')
        ax1.legend()
    
    # Tahmin bilgileri
    ax2.axis('off')
    
    pred_text = f"Model Tahminleri:\n"
    pred_text += f"PRI Modu: {predictions['pred_name'][0]}\n"
    if 'pri_us' in predictions:
        pri_value = predictions['pri_us'][0] if hasattr(predictions['pri_us'], '__len__') else predictions['pri_us']
        pred_text += f"Tahmin Edilen PRI: {pri_value:.2f} μs\n"
    if 'probs' in predictions:
        probs_array = predictions['probs'][0] if hasattr(predictions['probs'][0], '__len__') else predictions['probs']
        pred_text += f"Güven Skoru: {np.max(probs_array):.3f}\n"
    
    # Histogram istatistikleri
    max_hist_idx = np.argmax(hist)
    max_ipi = bin_centers[max_hist_idx]
    pred_text += f"\nHistogram İstatistikleri:\n"
    pred_text += f"En Yüksek Frekans IPI: {max_ipi:.2f}μs\n"
    pred_text += f"Histogram Maksimum Değeri: {hist.max():.4f}\n"
    
    # Tüm sınıfların olasılıklarını göster
    if 'probs' in predictions and 'classes' in predictions:
        probs_array = predictions['probs'][0] if hasattr(predictions['probs'][0], '__len__') else predictions['probs']
        pred_text += f"\nTüm Sınıf Olasılıkları:\n"
        for i, prob in enumerate(probs_array):
            class_name = predictions['classes'][i] if 'classes' in predictions else f"Class {i}"
            pred_text += f"  {class_name}: {prob:.3f}\n"
    
    ax2.text(0.1, 0.5, pred_text, transform=ax2.transAxes, 
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram görselleştirmesi kaydedildi: {save_path}")
    
    plt.show()

def visualize_predictions(X_val, y_val, y_logpri_val, predictions, classes, save_path=None, num_samples=20):
    """
    Validation set üzerinde ground truth ve predicted değerleri görselleştirir
    """
    # Rastgele örnekler seç
    indices = np.random.choice(len(X_val), min(num_samples, len(X_val)), replace=False)
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Ground truth bilgileri
        gt_class_idx = y_val[idx]
        gt_class_name = classes[gt_class_idx]
        gt_pri_us = np.exp(y_logpri_val[idx]) if y_logpri_val is not None else None
        
        # Predicted bilgileri
        pred_class_idx = predictions["pred_idx"][idx]
        pred_class_name = predictions["pred_name"][idx]
        pred_pri_us = predictions["pri_us"][idx] if "pri_us" in predictions else None
        
        # Metin bilgilerini hazırla
        info_text = f"Sample {idx}\n"
        info_text += f"GT Mode: {gt_class_name}\n"
        info_text += f"Pred Mode: {pred_class_name}\n"
        if gt_pri_us is not None and pred_pri_us is not None:
            info_text += f"GT PRI: {gt_pri_us:.2f}μs\n"
            info_text += f"Pred PRI: {pred_pri_us:.2f}μs"
        
        # Arka plan rengini belirle: GT ve Prediction aynıysa yeşil, değilse kırmızı
        background_color = 'lightgreen' if gt_class_idx == pred_class_idx else 'lightcoral'
        
        # Görselleştirme - arka plan rengi ile
        ax.text(0.1, 0.5, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor=background_color, edgecolor='black', linewidth=1))
        
        # Arka plan rengini ayarla
        ax.set_facecolor(background_color)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Kullanılmayan subplotları gizle
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('PRI Mode Classification Results\n(Green=Correct, Red=Incorrect)', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def infer(model_ckpt, X_feat):
    ck = torch.load(model_ckpt, map_location="cpu", weights_only=False)
    X = X_feat.astype(np.float32)
    # z-score
    mean = ck["mean"]; std = np.where(ck["std"] == 0, 1.0, ck["std"])
    X = (X - mean) / std

    model = PRINet(
        input_dim=ck["input_dim"],
        num_classes=len(ck["classes"]["classes"]),
        hidden_dims=tuple(int(v) for v in ck["hidden"].split(",") if v.strip()),
        dropout=ck["dropout"],
        do_regression=ck["do_regression"]
    )
    model.load_state_dict(ck["model_state"])
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(X)
        out = model(x)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=1).numpy()
        cls_idx = probs.argmax(axis=1)
        cls_name = [ck["classes"]["classes"][i] for i in cls_idx]
        result = {"probs": probs, "pred_idx": cls_idx, "pred_name": cls_name}
        if ck["do_regression"]:
            logpri = out["logpri_pred"].numpy().squeeze()
            pri_us = np.exp(logpri)
            result["pri_us"] = pri_us
    return result

def infer_with_validation(model_ckpt, save_plot=True, num_samples=20):
    """
    Validation kümesi ile inference yapar ve görselleştirir
    """
    ck = torch.load(model_ckpt, map_location="cpu", weights_only=False)
    
    # Validation set kontrolü
    if "X_val" not in ck or "y_val" not in ck:
        print("Error: Validation set not found in model checkpoint!")
        print("Please retrain the model to include validation set.")
        return None
    
    X_val = ck["X_val"]
    y_val = ck["y_val"]
    y_logpri_val = ck.get("y_logpri_val", None)
    classes = ck["classes"]["classes"]
    
    print(f"Validation set loaded: {X_val.shape[0]} samples")
    print(f"Classes: {classes}")
    
    # Inference
    predictions = infer(model_ckpt, X_val)
    
    # Accuracy hesapla
    accuracy = (predictions["pred_idx"] == y_val).mean()
    print(f"Validation Accuracy: {accuracy:.3f}")
    
    # Görselleştirme
    if save_plot:
        plot_path = Path(model_ckpt).parent / "validation_results.png"
        visualize_predictions(X_val, y_val, y_logpri_val, predictions, classes, 
                            save_path=str(plot_path), num_samples=num_samples)
    
    return predictions, X_val, y_val, y_logpri_val

def process_test_data(model_path, verici_num, frame_num, save_plot=True):
    """
    Test data klasöründen ham veri yükler, öznitelik çıkarır ve model tahmini yapar.
    
    Args:
        model_path: str - model dosyası yolu
        verici_num: int - verici numarası (2 veya 3)
        frame_num: int - frame numarası
        save_plot: bool - görselleştirmeyi kaydet
    """
    # Dosya yolu oluştur
    data_path = f"test data/single {verici_num}/deinterleaving_u_net_test_1_verici_2_scenario_single_{verici_num}_frame_{frame_num}.npy"
    
    if not os.path.exists(data_path):
        print(f"Hata: Dosya bulunamadı: {data_path}")
        return None
    
    print(f"Veri yükleniyor: {data_path}")
    
    # Ham veriyi yükle
    iq_data = np.load(data_path)
    print(f"Veri boyutu: {iq_data.shape}")
    
    # Öznitelik çıkar
    print("Öznitelik çıkarılıyor...")
    features_dict = extract_features_from_iq_data(iq_data)
    features = features_dict["feat_vector"]
    hist = features_dict["hist"]
    bin_edges_us = features_dict["bin_edges_us"]
    print(f"Öznitelik boyutu: {features.shape}")
    print(f"Histogram boyutu: {hist.shape}")
    
    # Model tahmini yap
    print("Model tahmini yapılıyor...")
    features_reshaped = features.reshape(1, -1)  # Batch dimension ekle
    predictions = infer(model_path, features_reshaped)
    
    # Sınıf isimlerini ekle
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    predictions['classes'] = ck["classes"]["classes"]
    
    # Sonuçları yazdır
    print(f"\n=== TAHMİN SONUÇLARI ===")
    print(f"Verici: {verici_num}")
    print(f"Frame: {frame_num}")
    print(f"PRI Modu: {predictions['pred_name'][0]}")
    if 'pri_us' in predictions:
        pri_value = predictions['pri_us'][0] if hasattr(predictions['pri_us'], '__len__') else predictions['pri_us']
        print(f"Tahmin Edilen PRI: {pri_value:.2f} μs")
    if 'probs' in predictions:
        probs_array = predictions['probs'][0] if hasattr(predictions['probs'][0], '__len__') else predictions['probs']
        print(f"Güven Skoru: {np.max(probs_array):.3f}")
        print(f"\nTüm Sınıf Olasılıkları:")
        for i, prob in enumerate(probs_array):
            print(f"  {predictions['classes'][i]}: {prob:.3f}")
    
    # Görselleştirme
    if save_plot:
        # Ham veri görselleştirmesi
        output_path_iq = f"test_result_iq_verici_{verici_num}_frame_{frame_num}.png"
        visualize_raw_data_with_predictions(iq_data, predictions, save_path=output_path_iq)
        
        # Histogram görselleştirmesi
        output_path_hist = f"test_result_hist_verici_{verici_num}_frame_{frame_num}.png"
        visualize_histogram(hist, bin_edges_us, predictions, save_path=output_path_hist)
    
    return predictions, iq_data, features_dict

def evaluate_folder_performance(model_path, folder_path, true_class_name, max_files=None):
    """
    Bir klasördeki tüm .npy dosyalarını (ham I/Q veri) kullanarak
    model performansını hesaplar. Klasördeki tüm örneklerin aynı sınıfa
    ait olduğu varsayılır.

    Args:
        model_path: str - model dosyası yolu
        folder_path: str - .npy dosyalarının bulunduğu klasör
        true_class_name: str - bu klasördeki tüm örneklerin gerçek sınıf adı
        max_files: int veya None - maksimum kaç dosya kullanılacak (opsiyonel)

    Returns:
        metrics: dict - accuracy ve özet bilgiler
    """
    # Klasördeki npy dosyalarını listele
    file_list = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
    if len(file_list) == 0:
        print(f"Hata: Klasörde .npy dosyası bulunamadı: {folder_path}")
        return None

    if max_files is not None:
        file_list = file_list[:max_files]

    print(f"Klasörden {len(file_list)} dosya bulundu: {folder_path}")

    # Tüm dosyalar için feature çıkar
    feat_list = []
    used_files = []
    for fpath in file_list:
        try:
            iq_data = np.load(fpath)
        except Exception as e:
            print(f"Dosya yüklenemedi, atlanıyor: {fpath} (hata: {e})")
            continue

        if iq_data.ndim != 2 or iq_data.shape[1] != 2:
            print(f"Uygun formatta olmayan veri, atlanıyor: {fpath} (shape={iq_data.shape})")
            continue

        try:
            features_dict = extract_features_from_iq_data(iq_data)
            feat = features_dict["feat_vector"]
            feat_list.append(feat)
            used_files.append(fpath)
        except Exception as e:
            print(f"Öznitelik çıkarılamadı, atlanıyor: {fpath} (hata: {e})")
            continue

    if len(feat_list) == 0:
        print("Hata: Hiçbir dosyadan öznitelik çıkarılamadı!")
        return None

    X_batch = np.stack(feat_list, axis=0)
    print(f"Toplam kullanılan örnek sayısı: {X_batch.shape[0]}")
    print(f"Öznitelik boyutu: {X_batch.shape[1]}")

    # Model ve sınıf bilgilerini yükle
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    classes = ck["classes"]["classes"]
    if true_class_name not in classes:
        print(f"Hata: Verilen gerçek sınıf adı model sınıfları içinde yok: {true_class_name}")
        print(f"Model sınıfları: {classes}")
        return None

    true_idx = classes.index(true_class_name)

    # Inference
    predictions = infer(model_path, X_batch)
    pred_idx = predictions["pred_idx"]
    pred_name = predictions["pred_name"]

    # Performans metriği: accuracy
    correct_mask = (pred_idx == true_idx)
    num_total = len(pred_idx)
    num_correct = int(correct_mask.sum())
    num_incorrect = num_total - num_correct
    accuracy = num_correct / num_total

    print("\n=== KLASÖR PERFORMANS ÖZETİ ===")
    print(f"Klasör: {folder_path}")
    print(f"Gerçek sınıf: {true_class_name} (idx={true_idx})")
    print(f"Toplam örnek: {num_total}")
    print(f"Doğru sınıflandırılan: {num_correct}")
    print(f"Hatalı sınıflandırılan: {num_incorrect}")
    print(f"Accuracy: {accuracy:.3f}")

    # Hatalı örnekleri özetle
    if num_incorrect > 0:
        print("\nHatalı örnekler (ilk 20):")
        wrong_indices = np.where(~correct_mask)[0]
        for i in wrong_indices[:20]:
            print(f"- File: {used_files[i]}")
            print(f"  Predicted: {pred_name[i]}")

    # Tahmin dağılımı
    print("\nTahmin sınıf dağılımı:")
    unique_idx, counts = np.unique(pred_idx, return_counts=True)
    for idx_val, cnt in zip(unique_idx, counts):
        cname = classes[idx_val] if idx_val < len(classes) else f"Class {idx_val}"
        print(f"  {cname}: {cnt} örnek")

    metrics = {
        "accuracy": accuracy,
        "num_total": num_total,
        "num_correct": num_correct,
        "num_incorrect": num_incorrect,
        "true_class": true_class_name,
        "classes": classes,
    }
    return metrics

def main():
    ap = argparse.ArgumentParser(description="PRI Modu Tahmin Sistemi")
    ap.add_argument("--model", type=str, required=True, help="Model dosyası yolu")
    ap.add_argument("--X", type=str, help="npy file with features (N, D)")
    ap.add_argument("--validation", action="store_true", help="Use validation set from model checkpoint")
    ap.add_argument("--test_data", action="store_true", help="Test data klasöründen ham veri işle")
    ap.add_argument("--verici", type=int, help="Verici numarası (2 veya 3)")
    ap.add_argument("--frame", type=int, help="Frame numarası")
    ap.add_argument("--num_samples", type=int, default=20, help="Number of samples to visualize")
    ap.add_argument("--save_plot", action="store_true", help="Save visualization plot")
    ap.add_argument("--folder", type=str, help="Performans ölçümü için .npy dosyaları içeren klasör")
    ap.add_argument("--folder_class", type=str, help="--folder içindeki örneklerin gerçek sınıf adı")
    ap.add_argument("--max_files", type=int, help="--folder için maksimum dosya sayısı (opsiyonel)")
    args = ap.parse_args()

    # 1) Klasör bazlı performans ölçümü modu
    if args.folder is not None:
        if args.folder_class is None:
            print("Hata: --folder kullanırken --folder_class argümanı da verilmelidir!")
            return
        metrics = evaluate_folder_performance(
            args.model,
            args.folder,
            args.folder_class,
            max_files=args.max_files,
        )
        if metrics is None:
            print("Klasör performans hesaplaması başarısız oldu.")
        return

    # 2) Test data modu (ham I/Q veri ile)
    if args.test_data:
        # Test data ile inference
        if args.verici is None or args.frame is None:
            print("Hata: --verici ve --frame argümanları gerekli!")
            print("Kullanım: python pri_infer.py --model model.pt --test_data --verici 2 --frame 100")
            return
        
        if args.verici not in [2, 3]:
            print("Hata: Verici numarası 2 veya 3 olmalı!")
            return
        
        result = process_test_data(args.model, args.verici, args.frame, save_plot=args.save_plot)
        if result is None:
            print("İşlem başarısız!")
            return
        
    elif args.validation:
        # Validation set ile inference
        result = infer_with_validation(args.model, save_plot=args.save_plot, num_samples=args.num_samples)
        if result is not None:
            predictions, X_val, y_val, y_logpri_val = result
            print("\nFirst 10 predictions:")
            for i in range(min(10, len(predictions["pred_name"]))):
                print(f"Sample {i}: {predictions['pred_name'][i]}")
                if "pri_us" in predictions:
                    print(f"  PRI: {predictions['pri_us'][i]:.2f}μs")
    else:
        # Normal inference
        if args.X is None:
            print("Error: --X argument is required when not using --validation or --test_data mode")
            return
        
        X = np.load(args.X)
        res = infer(args.model, X)
        print("pred_name:", res["pred_name"][:10])
        if "pri_us" in res:
            print("pri_us (first 10):", res["pri_us"][:10])

if __name__ == "__main__":
    main()
