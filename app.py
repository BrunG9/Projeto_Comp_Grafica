import cv2
from flask import Flask, render_template, Response, request, redirect, url_for

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

def get_video_capture():
    for index in range(3):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                print(f"Usando a câmera no índice {index} com backend {backend}")
                return cap
            else:
                print(f"Falha ao acessar a câmera no índice {index} com backend {backend}")
    return None

def detect_faces():
    cap = get_video_capture()
    if not cap:
        print("Erro: Não foi possível acessar a câmera.")
        return

    no_face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            no_face_count += 1
        else:
            no_face_count = 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        if no_face_count >= 3:
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n'
                   b'Desclassificado\r\n')
            no_face_count = 0
        else:
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/start_exam', methods=['POST'])
def start_exam():
    name = request.form['name']
    exam_name = request.form['exam_name']
    exam_number = request.form['exam_number']
    return render_template('index.html', name=name, exam_name=exam_name, exam_number=exam_number)

@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
