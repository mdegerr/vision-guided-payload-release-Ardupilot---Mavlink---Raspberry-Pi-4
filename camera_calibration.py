import cv2
import numpy as np
import glob

# Satranç tahtası iç köşe sayısı (yatay, dikey)
chessboard_size = (9, 6)  # 9 yatay, 6 dikey iç köşe
frame_size = (320, 240)   # Kullanacağınız çözünürlük

objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('calib_images/*.jpg')

print(f"{len(images)} adet fotoğraf bulundu. Köşe tespiti başlıyor...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)
    else:
        print(f"Köşe bulunamadı: {fname}")
cv2.destroyAllWindows()

if len(objpoints) < 10:
    print("UYARI: Yeterli sayıda başarılı köşe tespiti yok! Daha fazla ve farklı açılardan fotoğraf çekin.")
else:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)
    np.savez('calib.npz', mtx=mtx, dist=dist)
    print("Kamera kalibrasyon dosyası kaydedildi: calib.npz")
    print("Kullanılacak matris:")
    print(mtx)
    print("Distorsiyon katsayıları:")
    print(dist) 