#!/usr/bin/env python3
# -- coding: utf-8 --

"""
Kamera Test Scripti
Bu script kamera bağlantısını ve görüntü akışını test eder.
"""

import cv2
import numpy as np
import time
import sys

def test_frame_drop(cap, duration_sec=10):
    """Frame drop tespiti için test yapar"""
    print(f"🔍 {duration_sec} saniye boyunca frame drop tespiti yapılıyor...")

    frame_count = 0
    drop_count = 0
    start_time = time.time()
    expected_frame_count = 0
    last_frame_time = start_time

    print("   Frame drop analizi başlıyor...")
    print("   Çıkmak için herhangi bir tuşa basın")

    while (time.time() - start_time) < duration_sec:
        ret, frame = cap.read()

        current_time = time.time()
        expected_frame_count += 1

        if not ret:
            drop_count += 1
            print(f"⚠️ Frame drop! (Toplam: {drop_count})")
            time.sleep(0.01)  # Kısa bekleme
            continue

        frame_count += 1

        # Frame zamanlaması kontrolü
        frame_interval = current_time - last_frame_time
        expected_fps = cap.get(cv2.CAP_PROP_FPS)
        expected_interval = 1.0 / expected_fps if expected_fps > 0 else 0.033

        if frame_interval > expected_interval * 2:  # 2x beklenenden uzun
            print(f"⚠️ Frame gecikmesi: {frame_interval:.3f}s (beklenen: {expected_interval:.3f}s)")

        last_frame_time = current_time

        # Her 50 framede bir rapor
        if frame_count % 50 == 0:
            fps_current = frame_count / (current_time - start_time)
            drop_rate = (drop_count / expected_frame_count) * 100 if expected_frame_count > 0 else 0
            print(f"📊 Frame: {frame_count}, FPS: {fps_current:.1f}, Drop Rate: {drop_rate:.1f}%")

        # Klavye kontrolü (çıkış için)
        if cv2.waitKey(1) != -1:
            break

    # Sonuçları raporla
    total_time = time.time() - start_time
    final_fps = frame_count / total_time if total_time > 0 else 0
    final_drop_rate = (drop_count / expected_frame_count) * 100 if expected_frame_count > 0 else 0

    print("\n📈 Frame Drop Testi Sonuçları:")
    print(f"   Toplam Frame: {frame_count}")
    print(f"   Drop Sayısı: {drop_count}")
    print(f"   Drop Rate: {final_drop_rate:.2f}%")
    print(f"   Ortalama FPS: {final_fps:.1f}")
    print(f"   Test Süresi: {total_time:.1f}s")

    if final_drop_rate > 5:
        print("❌ KRİTİK: Yüksek frame drop rate'i!")
        print("   Çözüm önerileri:")
        print("   - USB 3.0 portu kullanın")
        print("   - Çözünürlüğü düşürün (320x240)")
        print("   - FPS'yi düşürün (15-20)")
        print("   - USB extender kullanmayın")
        print("   - Diğer USB cihazları çıkarın")
    elif final_drop_rate > 1:
        print("⚠️ ORTA: Hafif frame drop var")
        print("   - Çözünürlük veya FPS'yi düşürmeyi deneyin")
    else:
        print("✅ İYİ: Frame drop problemi yok")

    return final_drop_rate, final_fps

def test_camera(camera_index=0, width=320, height=240):
    """Belirtilen kamera index'i ile kamera testi yapar - Frame Drop tespiti ile"""

    print(f"🔍 Kamera {camera_index} test ediliyor...")
    print(f"   Çözünürlük: {width}x{height}")

    # Kamerayı aç
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ Kamera {camera_index} açılamadı!")
        print("   Çözümler:")
        print("   - Kameranın bağlı olduğunu kontrol edin")
        print("   - USB portunu değiştirin")
        print("   - Başka kamera uygulamalarını kapatın")
        print(f"   - Farklı kamera index'i deneyin (örnek: {camera_index+1})")
        return False

    # Kamera özelliklerini ayarla
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Kamera bilgilerini al
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("✅ Kamera açıldı!")
    print(f"   Gerçek çözünürlük: {actual_width}x{actual_height}")
    print(f"   FPS: {fps}")

    # Test frame oku
    print("📸 Test frame okunuyor...")
    ret, test_frame = cap.read()

    if ret and test_frame is not None:
        print("✅ Test frame başarılı!")
        print(f"   Frame boyutu: {test_frame.shape}")
        print(f"   Frame tipi: {test_frame.dtype}")

        # Frame'in siyah olup olmadığını kontrol et
        if np.sum(test_frame) == 0:
            print("⚠️ UYARI: Frame tamamen siyah!")
        else:
            print("✅ Frame renk bilgisi içeriyor")
    else:
        print("❌ Test frame okunamadı!")
        cap.release()
        return False

    # Önce frame drop testi yap
    print("\n🔬 Frame drop analizi başlıyor...")
    drop_rate, measured_fps = test_frame_drop(cap, duration_sec=5)  # 5 saniye test

    # Canlı görüntü gösterimi
    print("\n🎥 Canlı görüntü test ediliyor...")
    print("   Çıkmak için 'q' tuşuna basın")
    print("   Kayıt için 's' tuşuna basın")
    print("   Frame drop testi için 'd' tuşuna basın")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("⚠️ Frame okunamadı, devam ediliyor...")
            time.sleep(0.01)
            continue

        frame_count += 1

        # Her 30 framede bir bilgi göster
        if frame_count % 30 == 0:
            fps_current = frame_count / (time.time() - start_time)
            print(f"📊 Frame: {frame_count}, FPS: {fps_current:.1f}")

        # Frame'i göster
        cv2.imshow(f'Kamera Test - Index {camera_index}', frame)

        # Klavye kontrolü
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("👋 Çıkış yapılıyor...")
            break
        elif key == ord('s'):
            # Ekran görüntüsü kaydet
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"camera_test_{camera_index}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Ekran görüntüsü kaydedildi: {filename}")
        elif key == ord('d'):
            # Frame drop testi tekrar
            print("\n🔄 Frame drop testi tekrar başlatılıyor...")
            test_drop_rate, test_fps = test_frame_drop(cap, duration_sec=3)
            print(f"🔍 Güncel sonuçlar - Drop Rate: {test_drop_rate:.2f}%, FPS: {test_fps:.1f}")

    # Temizlik
    cap.release()
    cv2.destroyAllWindows()

    print("✅ Kamera testi tamamlandı!")
    print("📈 Final Sonuçlar:")
    print(f"   - Frame Drop Rate: {drop_rate:.2f}%")
    print(f"   - Ortalama FPS: {measured_fps:.1f}")
    return True

def main():
    """Ana fonksiyon"""
    print("🎥 Kamera Test Uygulaması")
    print("=" * 40)

    # Komut satırı argümanlarını kontrol et
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print("❌ Geçersiz kamera index'i!")
            print("   Kullanım: python camera_test.py [kamera_index] [genişlik] [yükseklik]")
            return
    else:
        camera_index = 0

    # Çözünürlük parametrelerini al
    width = 320
    height = 240

    if len(sys.argv) > 2:
        try:
            width = int(sys.argv[2])
        except ValueError:
            pass

    if len(sys.argv) > 3:
        try:
            height = int(sys.argv[3])
        except ValueError:
            pass

    # Kamerayı test et
    success = test_camera(camera_index, width, height)

    if not success:
        print("\n🔄 Farklı kamera index'leri deneniyor...")

        # Otomatik olarak farklı index'leri dene
        for i in range(1, 5):
            if i == camera_index:
                continue

            print(f"\n🔍 Kamera {i} deneniyor...")
            if test_camera(i, width, height):
                break

    print("\n💡 İpuçları:")
    print("   - Kameranız çalışmıyorsa USB portunu değiştirin")
    print("   - Raspberry Pi kullanıyorsanız 'vcgencmd get_camera' komutunu deneyin")
    print("   - Linux'ta 'v4l2-ctl --list-devices' ile kamera listesini görün")

if __name__ == "__main__":
    main()
