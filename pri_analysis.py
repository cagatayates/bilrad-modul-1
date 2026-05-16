import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def load_and_visualize_iq_data():
    """
    Single data klasöründeki tüm .npy dosyalarını yükler ve I/Q verilerini görselleştirir.
    Her dosya için ayrı grafik oluşturur ve I, Q, magnitude değerlerini gösterir.
    """
    
    # Single data klasöründeki tüm .npy dosyalarını bul
    data_folder = "single data"
    npy_files = glob.glob(os.path.join(data_folder, "*.npy"))
    
    if not npy_files:
        print("Single data klasöründe .npy dosyası bulunamadı!")
        return
    
    print(f"Bulunan dosya sayısı: {len(npy_files)}")
    
    # Her dosya için ayrı grafik oluştur
    for i, file_path in enumerate(npy_files):
        print(f"\nİşleniyor: {os.path.basename(file_path)}")
        
        # Veriyi yükle
        try:
            data = np.load(file_path)
            print(f"Veri boyutu: {data.shape}")
            
            # I ve Q verilerini ayır
            I_data = data[:, 0]  # İlk sütun I (in-phase)
            Q_data = data[:, 1]  # İkinci sütun Q (quadrature)
            
            # Magnitude hesapla: |z| = sqrt(I^2 + Q^2)
            magnitude = np.sqrt(I_data**2 + Q_data**2)
            
            # Grafik oluştur
            plt.figure(figsize=(12, 8))
            
            # Zaman ekseni oluştur
            time_axis = np.arange(len(I_data))
            
            # I, Q ve magnitude'ı aynı grafikte göster
            plt.subplot(3, 1, 1)
            plt.plot(time_axis, I_data, 'b-', linewidth=0.5, alpha=0.7, label='I (In-phase)')
            plt.title(f'I/Q Verisi - {os.path.basename(file_path)}')
            plt.ylabel('I Değeri')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.subplot(3, 1, 2)
            plt.plot(time_axis, Q_data, 'r-', linewidth=0.5, alpha=0.7, label='Q (Quadrature)')
            plt.ylabel('Q Değeri')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.subplot(3, 1, 3)
            plt.plot(time_axis, magnitude, 'g-', linewidth=0.5, alpha=0.7, label='Magnitude')
            plt.xlabel('Zaman (Sample)')
            plt.ylabel('Magnitude')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            
            # Grafik dosyasını kaydet
            output_filename = f"iq_analysis_{os.path.splitext(os.path.basename(file_path))[0]}.png"
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {output_filename}")
            
            # İstatistikleri yazdır
            print(f"I verisi - Min: {np.min(I_data):.4f}, Max: {np.max(I_data):.4f}, Ortalama: {np.mean(I_data):.4f}")
            print(f"Q verisi - Min: {np.min(Q_data):.4f}, Max: {np.max(Q_data):.4f}, Ortalama: {np.mean(Q_data):.4f}")
            print(f"Magnitude - Min: {np.min(magnitude):.4f}, Max: {np.max(magnitude):.4f}, Ortalama: {np.mean(magnitude):.4f}")
            
            # Grafik göster (isteğe bağlı)
            plt.show()
            
        except Exception as e:
            print(f"Hata: {file_path} dosyası yüklenirken hata oluştu: {e}")
            continue

def create_combined_visualization():
    """
    Tüm dosyaları tek bir grafikte karşılaştırmalı olarak gösterir.
    """
    data_folder = "single data"
    npy_files = glob.glob(os.path.join(data_folder, "*.npy"))
    
    if not npy_files:
        print("Single data klasöründe .npy dosyası bulunamadı!")
        return
    
    plt.figure(figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, file_path in enumerate(npy_files):
        try:
            data = np.load(file_path)
            I_data = data[:, 0]
            Q_data = data[:, 1]
            magnitude = np.sqrt(I_data**2 + Q_data**2)
            
            color = colors[i % len(colors)]
            filename = os.path.basename(file_path)
            
            # I verisi
            plt.subplot(3, 1, 1)
            plt.plot(I_data, color=color, linewidth=0.5, alpha=0.7, label=f'I - {filename}')
            
            # Q verisi
            plt.subplot(3, 1, 2)
            plt.plot(Q_data, color=color, linewidth=0.5, alpha=0.7, label=f'Q - {filename}')
            
            # Magnitude
            plt.subplot(3, 1, 3)
            plt.plot(magnitude, color=color, linewidth=0.5, alpha=0.7, label=f'Magnitude - {filename}')
            
        except Exception as e:
            print(f"Hata: {file_path} dosyası yüklenirken hata oluştu: {e}")
            continue
    
    # Grafik ayarları
    plt.subplot(3, 1, 1)
    plt.title('Tüm Dosyalar - I/Q Verisi Karşılaştırması')
    plt.ylabel('I Değeri')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplot(3, 1, 2)
    plt.ylabel('Q Değeri')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplot(3, 1, 3)
    plt.xlabel('Zaman (Sample)')
    plt.ylabel('Magnitude')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("combined_iq_analysis.png", dpi=300, bbox_inches='tight')
    print("Birleşik grafik kaydedildi: combined_iq_analysis.png")
    plt.show()

if __name__ == "__main__":
    print("I/Q Veri Analizi Başlatılıyor...")
    print("=" * 50)
    
    # Her dosya için ayrı grafik oluştur
    load_and_visualize_iq_data()
    
    print("\n" + "=" * 50)
    print("Birleşik karşılaştırma grafiği oluşturuluyor...")
    
    # Tüm dosyaları tek grafikte göster
    create_combined_visualization()
    
    print("\nAnaliz tamamlandı!")
