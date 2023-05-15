# BOLUM 1: Gerekli kütüphanelerin yüklenmesi 

import numpy as np 
import cv2


# BOLUM 2: Renklendirme için gerekli veri modelleri dosyalarının yüklenmesi 

print("Model Yükleniyor .....")
net = cv2.dnn.readNetFromCaffe('./model/colorization_deploy_v2.prototxt','./model/colorization_release_v2.caffemodel')
pts = np.load('./model/pts_in_hull.npy')



# BOLUM 3: Bu satırlar, class8_ab ve conv8_313_rh katmanlarının idlerini alarak  renklendirme modelindeki parametreleri bu katmanlara yükler.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2,313,1,1) # LAB renk uzayı için yeniden boyutlandırma yapılır



# BOLUM 4: Sinir ağının ağırlıklarını .blobs ile sinir ağının birinci ve ikinci katmanına ağırlıkları ekler
net.getLayer(class8).blobs = [pts.astype("float32")] # class8 değişkenine float32 tipine dönüştürerek ağırlık olarak ekler.
net.getLayer(conv8).blobs = [np.full([1,313],2.606,dtype='float32')] # 1x313 boyutunda 2.606 değerleri ile bir dizi ağırlık olarak atanır.



# BOLUM 5: Resimi okuma , ölçeklendirme ve LAB uzayına çevirme
image = cv2.imread('./images/resim4.jpg') # resim okunur.
scaled = image.astype("float32")/255.0 # float32 formatına dönüştürüp 0-1 arasında değerler ile ölçeklendirilir.
lab = cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB) # ölçeklendirilmiş fotoğraf BGR formatından LAB formatına dönüştürülür.


# BOLUM 6: LAB formatının üzerinde işlem yapılıyor
resized = cv2.resize(lab,(224,224))
L = cv2.split(resized)[0]
L -= 50 # LAB formatının ilk değeri olan L "lightness" katmanını L değerine atayıp değerden 50 çıkartılıyor


# BOLUM 7: Renklendirme kısmı

net.setInput(cv2.dnn.blobFromImage(L)) # Gri tonlamalı L kanalı alınır ve dnn.blobFromImage fonksiyonu ile sinir ağına uygun girdi oluşur.

ab = net.forward()[0, :, :, :].transpose((1,2,0)) # Modelden forward() fonksiyonu ile "AB" kanalı için uygun çıktı alınır

ab = cv2.resize(ab, (image.shape[1],image.shape[0])) #  Tahmin edilmiş "AB" verileri ile görüntü orjinal boyuta göre yeniden boyutlanır.

L = cv2.split(lab)[0] # Görüntüyü "L" , "AB" kısımlarına ayrılır

colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2) # Numpy kütüphanesi ile concatenate() fonksiyonu ile "L" ile "AB" kanalını birleştirir.
# np.newaxis parametresi, L kanalının boyutunu 2D'den 3D'ye çıkarmak için kullanılır. 



colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR) # LAB formatından tekrardan BGR formatına dönüştürülüp colorized değişkenine aktarılır

colorized = (255 * colorized).astype("uint8") # float32 formatı ile 0-1 arasında değerlerle atanmış diziyi 255 ile çarparak 0-255 
# arasında tam sayı tipine dönüştürülür. 


#BOLUM  8: Renklendirilmiş ve Siyah-Beyaz fotoğrafların karşılaştırılması 
cv2.imshow("Original",image)
cv2.imshow("Colorized",colorized)

cv2.waitKey(0)