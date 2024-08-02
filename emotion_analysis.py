import cv2
from fer import FER

def analyze_emotions_in_real_time():
    """
    Запускает анализ эмоций в реальном времени с использованием веб-камеры.
    Распознаёт лица и определяет доминирующую эмоцию, отображая её на видео в режиме реального времени.
    """
    # Инициализация захвата видео с веб-камеры (индекс 0)
    cap = cv2.VideoCapture(0)
    detector = FER()

    if not cap.isOpened():
        print("Не удалось открыть веб-камеру.")
        return

    print("Анализ эмоций начат. Нажмите 'q', чтобы выйти.")

    while True:
        # Чтение текущего кадра с веб-камеры
        ret, frame = cap.read()

        if not ret:
            print("Не удалось получить кадр. Выход из программы.")
            break

        try:
            # Анализ эмоций на текущем кадре
            analysis = detector.detect_emotions(frame)

            # Обработка каждого обнаруженного лица
            for face in analysis:
                (x, y, w, h) = face['box']
                emotions = face['emotions']
                dominant_emotion = max(emotions, key=emotions.get)

                # Обрамление лица и отображение доминирующей эмоции
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            print(f"Ошибка во время анализа: {e}")

        # Отображение кадра с результатами анализа
        cv2.imshow('EmotionVision by ROVA', frame)

        # Прерывание цикла при нажатии 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Завершение анализа по запросу пользователя.")
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()
    print("Анализ завершён и ресурсы освобождены.")

# Запуск функции анализа эмоций
if __name__ == "__main__":
    analyze_emotions_in_real_time()
