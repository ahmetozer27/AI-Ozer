# YOLOv11 + MediaPipe + Vosk AI Projesi

Gerçek zamanlı nesne tespiti, el hareketleri algılama ve sesli komut tanıma içeren Python tabanlı yapay zeka uygulaması.

---

## Özellikler

- **YOLOv11** ile nesne tespiti (el, yüz vb.)  
- **MediaPipe Hands** ile el parmak eklemlerinin gerçek zamanlı algılanması ve çizimi  
- **Vosk** kullanarak Türkçe sesli komutların tanınması  
- **PyAudio** ile mikrofon girişinden canlı ses yakalama  
- GPU hızlandırması ile yüksek performans  
- 60 FPS video yakalama ve işleme  
- Çoklu iş parçacığı (thread) ile ses ve video akışlarının eşzamanlı çalışması  

---

## Gereksinimler

- Python 3.7 ve üzeri  
- torch  
- ultralytics  
- opencv-python  
- mediapipe  
- vosk  
- pyaudio  

---

## Kurulum

```bash
pip install torch ultralytics opencv-python mediapipe vosk pyaudio
```

> **Not:**  
> Vosk için Türkçe modelini [buradan](https://alphacephei.com/vosk/models) indirip, model klasör yolunu kodda belirtmeniz gerekir.

---

## Kullanım

Aşağıdaki örnekle programı başlatabilirsiniz:

```python
from your_module_name import startAI  # Kodunu modüle dönüştürdüğünde

if __name__ == "__main__":
    startAI()
```

---

## Kod Akışı

1. **YOLOv11 Model Yükleme:**  
   `yolo11x.pt` modeli `ultralytics.YOLO` sınıfı ile yüklenir. Bu model video karelerinde nesne tespiti yapar.

2. **El Hareketleri Algılama:**  
   MediaPipe `Hands` modülü ile her karede elin parmak eklemleri tespit edilir ve çizilir.

3. **Sesli Komut Tanıma:**  
   Vosk modeli mikrofon verisinden gerçek zamanlı sesli komutları algılar.  
   Komutlar ayrı bir thread’de işlenir.

4. **Performans Optimizasyonu:**  
   CUDA destekliyorsa GPU otomatik kullanılır.  
   Kamera FPS 60 olarak ayarlanır.  
   Video ve ses işleme eşzamanlı ve akıcıdır.

5. **Uygulama Çıkışı:**  
   Kullanıcı 'q' tuşuna bastığında uygulama sonlanır.

---

## Örnek Komutlar

- "Merhaba" dediğinizde konsola `"Merhaba dediniz!"` yazılır.  
- "Çık", "kapat", "kapan" komutları uygulamayı sonlandırır.

---

## İletişim

Ahmet Özer  
ahmet@example.com

---

© 2025 Ahmet Özer  
