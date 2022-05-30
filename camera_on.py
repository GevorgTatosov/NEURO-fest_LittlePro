import cv2
import numpy as np
import onnxruntime as ort


def center_crop(frame):
    h, w, _ = frame.shape
    start = abs(h - w) // 2
    if h > w:
        return frame[start: start + w]
    return frame[:, start: start + h]


def main():
    words = ['Устаю', 'Болит', 'Быстро', 'Горло', 'Я(у меня)', 'Высокая', 'Температура']
    mean = 0.485 * 255.
    std = 0.229 * 255.

    ort_session = ort.InferenceSession("signlanguage.onnx")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        frame = center_crop(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(frame, (28, 28))
        x = (x - mean) / std

        x = x.reshape(1, 1, 28, 28).astype(np.float32)
        y = ort_session.run(None, {'input': x})[0]

        index = np.argmax(y, axis=1)

        print(words[int(index)])
        cv2.imshow("Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
