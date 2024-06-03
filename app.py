import cv2
from flask import Flask, render_template, Response, request, jsonify
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)

# Inicialização da captura de vídeo
cap = cv2.VideoCapture(0)

# Variáveis para contagem de frames
total_frames = 0
face_detected_frames = 0
no_face_detected_frames = 0
warning_count = 0

# Carrega o classificador de rosto
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/start_exam', methods=['POST'])
def start_exam():
    global name, exam_name, exam_number
    name = request.form['name']
    exam_name = request.form['exam_name']
    exam_number = request.form['exam_number']
    return render_template('index.html', name=name, exam_name=exam_name)

def generate_frames():
    global total_frames, face_detected_frames, no_face_detected_frames

    while True:
        success, frame = cap.read()
        if not success:
            print("Erro: Não foi possível capturar o frame.")
            break

        total_frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            face_detected_frames += 1
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        else:
            no_face_detected_frames += 1

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/send_warning', methods=['POST'])
def send_warning():
    global warning_count
    warning_count += 1
    if warning_count >= 3:
        return jsonify({'status': 'disqualified'})
    return jsonify({'status': 'warning', 'count': warning_count})

@app.route('/report')
def report():
    detection_rate = (face_detected_frames / total_frames) * 100 if total_frames > 0 else 0
    non_detection_rate = (no_face_detected_frames / total_frames) * 100 if total_frames > 0 else 0

    return render_template('report.html', 
                           detection_rate=detection_rate, 
                           non_detection_rate=non_detection_rate, 
                           total_frames=total_frames,
                           face_detected_frames=face_detected_frames,
                           no_face_detected_frames=no_face_detected_frames)

@app.route('/plot.png')
def plot_png():
    fig, ax = plt.subplots()
    labels = ['Face Detectada', 'Face Não Detectada']
    sizes = [face_detected_frames, no_face_detected_frames]
    colors = ['#ff9999','#66b3ff']
    
    # Garantir que não tentaremos dividir por zero
    if sum(sizes) > 0:
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    else:
        ax.pie([1], labels=['Nenhum dado'], colors=['#d3d3d3'], autopct='%1.1f%%', startangle=140)

    ax.axis('equal')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)  # Fechar a figura para liberar memória
    return Response(img.getvalue(), mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
