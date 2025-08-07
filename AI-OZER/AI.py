import cv2
import torch
import torch.version
from ultralytics import YOLO
import mediapipe as mp
from vosk import Model, KaldiRecognizer
import pyaudio
import json
import threading

def ahmet():
    print("--- AHMET ADAMDIR! ---")

# Ses tanıma fonksiyonu
def listen():
    model = Model("D:\\Programlama Projeleri\\Python Projeleri\\AI-Ozer\\vosk-model-small-tr-0.3\\")  # Türkçe model yolunu buraya verin
    recognizer = KaldiRecognizer(model, 16000)
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=4000)
    stream.start_stream()

    print("Konuşmaya başlayın...")
    
    while True:
        data = stream.read(4000)
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            command = json.loads(result)["text"]
            print(f"Komut: {command}")
            return command
        else:
            partial_result = recognizer.PartialResult()
            print(json.loads(partial_result)["partial"])

    stream.stop_stream()
    stream.close()
    p.terminate()

# Ses tanımayı farklı bir thread'de çalıştırmak için
def listen_thread():
    while True:
        command = listen()
        # Komutla ne yapılacaksa burada yapılabilir
        if "merhaba" in command.lower():
            print("Merhaba dediniz!")
        elif "çık" or "kapat" or "kapan" in command.lower():
            exit(-1)


def startAI():
    # Yüz ifadesi tanımak için model yükleyin
    # YOLOv5 modelini yükle (yolov5s en küçük modeldir)
    #model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
    model = YOLO('yolo11x.pt')

    # Yüz ifadesi etiketleri
    #expressions = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    # OpenCV yüz tespiti için Haar Cascade
    #face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

     # Cihazı kontrol et ve GPU kullanımı için ayar yap
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Device being used: {device}")
    print(torch.cuda.is_available())  # True dönerse CUDA kullanılabilir
    print(torch.version.cuda)         # PyTorch'un desteklediği CUDA sürümü
    print(torch.backends.cudnn.version())  # CuDNN sürümü
    print(torch.cuda.device_count())   # Kullanılabilir GPU sayısını gösterir
    print(torch.cuda.current_device()) # Şu anda kullanılan GPU'yu gösterir

    # El hareketi tanıma için MediaPipe başlat
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils


    # Kamerayı başlat
    cap = cv2.VideoCapture(0)
    target_fps = 60
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    # Ses dinleme için ayrı bir thread başlat
    thread = threading.Thread(target=listen_thread, daemon=True)
    thread.start()


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Modeli kullanarak nesne tespiti yap
        results = model(frame)

        # Sonuçları görüntüle
        frame = results[0].plot()  # Nesneleri işaretler

        # MediaPipe ile el hareketlerini algıla
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(rgb_frame)

        if results_hands.multi_hand_landmarks:
            for landmarks in results_hands.multi_hand_landmarks:
                # Elin işaretlerini çiz
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Sonuçları ekranda göster
        cv2.imshow('YOLOv11 - Web Kamera', frame)

        # 'q' tuşuna basıldığında çıkış yap
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

startAI()