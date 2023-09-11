from flask import Flask, render_template, url_for, Response, request
from flask_mysqldb import MySQL
import cv2

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'apnr_log'

mysql = MySQL(app)

# cam = cv2.VideoCapture("static/img/Video.mp4")
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def generate_frames():
    while True:
        ret, frame = cam.read()

        if not ret:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

        yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    try:
        cursor = mysql.connection.cursor()

        cursor.execute(''' SELECT * FROM `log` ORDER BY `id_log` DESC LIMIT 5''')
        result = cursor.fetchall()

        cursor.close()
        
        return render_template('index.html')
    except:
        return 'Error'
    
    

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/login', methods = ['POST', 'GET'])
def login():
    if request.method == 'GET':
        return "Please open login form"
    
    if request.method == "POST":
        nomor_kendaraan = request.form['nomor_kendaraan']
        waktu_masuk = request.form['waktu_masuk']
        waktu_keluar = request.form['waktu_keluar']
        pemilik = request.form['pemilik']
        status_kendaraan = request.form['status_kendaraan']

        try:
            cursor = mysql.connection.cursor()

            # cursor.execute(''' INSERT INTO `log`(`nomor_kendaraan`, `waktu_masuk`, `waktu_keluar`, 
            #        `pemilik`, `status_kendaraan`) 
            #        VALUES (%s, %s, %s, %s, %s) 
            #        ''', (nomor_kendaraan, waktu_masuk, waktu_keluar, pemilik, status_kendaraan))

            cursor.execute(''' SELECT * FROM `log` ''')
            result = cursor.fetchall()
            date = result[1][2]
            result = date.strftime("%Y-%m-%d %H:%M:%S")

            # mysql.connection.commit()
            cursor.close()
            return str(result)
        except:
            return 'Error'

if __name__ == "__main__":
    app.run(debug=True)
    # app.run()
