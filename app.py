from cv2 import data
from werkzeug.exceptions import HTTPException
from flask import request, Response, render_template_string, render_template
import flask
import jsonpickle
import numpy as np
import cv2
import os
import face_recognition
import logging
import os.path
import base64
import time
import pymysql
import dlib
import sys
import requests
import imutils

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

app = flask.Flask(__name__)


UPLOAD_FOLDER = os.path.join(app.root_path, 'upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 100, 'pool_recycle': 280}
app.config["mydb"] = None

def open_connection():
    try:
        if app.config["mydb"] is None:
            app.config["mydb"] = pymysql.connect(
                host=os.environ.get("DB_HOST"),
                user=os.environ.get("DB_USER"),
                password=os.environ.get("DB_PASS"),
                database="faceid"
            )
    except pymysql.MySQLError as e:
        logging.error(e)
        sys.exit()
    finally:
        logging.basicConfig(filename='error.log', level=logging.DEBUG)
        logging.info('Connection opened successfully.')


@app.errorhandler(Exception)
def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    response = {'message': 'Something went wrong, Please Try again'}
    data = jsonpickle.encode(response)
    logging.basicConfig(filename='error.log', level=logging.DEBUG)
    logging.warning(e)
    return Response(response=data, status=code, mimetype="application/json")


def run_query(query, data):
    try:
        open_connection()
        with app.config["mydb"].cursor() as cur:
            if 'SELECT' in query:
                records = []
                cur.execute(query)
                result = cur.fetchall()
                for row in result:
                    records.append(row)
                cur.close()
                return records
            result = cur.execute(query, data)
            app.config["mydb"].commit()
            cur.close()
            return result
    except pymysql.MySQLError as e:
        logging.basicConfig(filename='error.log', level=logging.DEBUG)
        logging.warning(e)
        sys.exit()
    finally:
        if app.config["mydb"]:
            app.config["mydb"].close()
            app.config["mydb"] = None
            logging.basicConfig(filename='error.log', level=logging.DEBUG)
            logging.info('Database connection closed.')

def check_if_limit_exceeded(table, mspin):
    sql= "SELECT * from "+table+" where mspin = "+mspin+" "
    result = run_query(sql, None)
    if(len(result) >= 10):
        if(table == 'registered_images'):
            delete_sql = "DELETE FROM result WHERE registered_img_id = "+str(result[0][0])+""
            run_query(delete_sql, None)
        delete_sql = "DELETE FROM "+table+" WHERE id = "+str(result[0][0])+""
        run_query(delete_sql, None)

def add_user_image_details(table, data):
    check_if_limit_exceeded(table, data[0])
    sql = "INSERT INTO "+table+" (mspin, path) VALUES (%s, %s)"
    return run_query(sql, data)


def update_result(data):
    check_if_limit_exceeded('result', data[0])
    sql = "INSERT INTO result (mspin, registered_img_id, verified_path, result, error) VALUES (%s, %s, %s, %s, %s)"
    return run_query(sql, data)


def registered_details(username):
    sql = "SELECT * FROM registered_images where mspin = " + \
        username+" order by id DESC limit 1"
    return run_query(sql, None)


def get_result(username):
    sql = "SELECT * FROM result where mspin = "+username+" "
    result = run_query(sql, None)
    return result


def get_results(username):
    sql = "SELECT res.id, res.mspin, ri.path, res.verified_path, res.result, res.error, res.added_date FROM result res JOIN registered_images ri ON res.registered_img_id = ri.id order by res.id DESC"
    if username:
        sql = "SELECT res.id, res.mspin, ri.path, res.verified_path, res.result, res.error, res.added_date FROM result res JOIN registered_images ri ON res.registered_img_id = ri.id where res.mspin = " + \
            username+" order by res.id DESC LIMIT 10"

    result = run_query(sql, None)
    return result


def read_image(file_path):
    return cv2.imread(file_path)


def face_info(file_path):
    result = False
    known_face = []
    known_image = read_image(file_path)
    data = find_face(known_image)
    known_encoding = data['face_encoding']
    known_image = data['valid_image']
    if len(known_encoding) > 0:
        # Load the detector
        detector = dlib.get_frontal_face_detector()
        gray = cv2.cvtColor(known_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y , w ,h) in faces:
            cv2.rectangle(known_image, (x,y), (x+w, y+h), (255, 0 , 0), 3)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = known_image[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                logging.info("eyes found")
                result = True
                break

        # Use detector to find landmarks
        known_face = detector(gray)
    
    output = dict()
    output['known_encoding'] = known_encoding
    output['known_face'] = known_face
    output['eyes'] = result
    return output

def find_face(image):
    known_encoding = face_recognition.face_encodings(image)
    logging.basicConfig(filename='error.log', level=logging.DEBUG)
    data = dict()
    data['face_encoding'] = known_encoding
    data['valid_image'] = image
    if len(known_encoding) == 0:
        for x in range(3):
            rotated_image = imutils.rotate(image, angle=90)
            logging.info("rotated 90 degree")
            image = rotated_image
            known_encoding = face_recognition.face_encodings(image)
            data['valid_image'] = image
            data['face_encoding'] = known_encoding
            if len(known_encoding) > 0:
                logging.info("Face encoding found after iteration")
                return data
            if x == 2:
                logging.info("No Face found after rotating")
                return data
    else:
        logging.info("Face encoding found before iteration")
        return data


@app.route('/')
def index():
    '''Index page route'''

    return '<h1>Application Deployed!</h1>'

# app.config["DEBUG"] = True


@app.route("/add_face", methods=["POST"])
def Add():

    logging.basicConfig(filename='error.log', level=logging.DEBUG)
    logging.info("****************START LOGS FOR REGISTER API****************")
    
    params_error = False
    if 'username' not in request.form:
        params_error = True
    if 'img' not in request.form:
        params_error = True

    if not params_error:
        username = request.form.get('username')
        logging.info("MSPIN ------>" + username + "<---------")
        user_image = request.form.get('img')
        imgdata = base64.b64decode(user_image)
        ts = time.time()
        image_name = str(round(ts))
        filename = os.path.join(
            app.config['UPLOAD_FOLDER'], image_name + ".png")
        with open(filename, 'wb') as f:
            f.write(imgdata)

        face_result = face_info(filename)
        
        if len(face_result["known_face"]) > 0 and len(face_result["known_encoding"]) > 0 and face_result["eyes"]:
            if len(face_result["known_face"]) > 1 or len(face_result["known_encoding"]) > 1:
                response_pickled = jsonpickle.encode({'message': 'Multiple Faces detected. Please try again.'})
                logging.info("****************END LOGS FOR REGISTER API****************")
                return Response(response=response_pickled, status=200, mimetype="application/json")

            else:
                response = {'filename': username+ ".png", 'message': 'file uploaded successfully'}
                response_pickled = jsonpickle.encode(response)
                add_user_image_details('registered_images',[username, filename])
                rest_params = {"mspin":username, "is_registered":1}
                logging.info("****************END LOGS FOR REGISTER API****************")
                response = requests.post(url="https://ilearnservice.marutisuzuki.com/import_user_data/face_register_callback.php",data=rest_params)
                return Response(response=response_pickled, status=200, mimetype="application/json")
        else:
            response = {'message': 'No Valid Face found, Please re-register'}
            response_pickled = jsonpickle.encode(response)
            logging.info("****************END LOGS FOR REGISTER API****************")
            return Response(response=response_pickled, status=200, mimetype="application/json")

    if params_error:
        response = {'message': 'invalid arguments'}
        data = jsonpickle.encode(response)
        logging.info("****************END LOGS FOR REGISTER API****************")
        return Response(response=data, status=400, mimetype="application/json")


@app.route("/verify", methods=["POST"])
def Verify():
    params_error = False
    if 'username' not in request.form:
        params_error = True
    if 'img' not in request.form:
        params_error = True

    logging.basicConfig(filename='error.log', level=logging.DEBUG)
    logging.info("****************START LOGS FOR VERIFY API****************")
    if not params_error:
        user_image = request.form.get('img')
        username = request.form.get('username')
        logging.info("MSPIN ------>" + username + "<---------")
        imgdata = base64.b64decode(user_image)

        user_info = registered_details(username)
        if user_info:
            ts = time.time()
            image_name = str(round(ts))
            new_path = os.path.join(
                app.config['UPLOAD_FOLDER'], image_name + ".png")
            error_info = ''
            with open(new_path, 'wb') as f:
                f.write(imgdata)

            known_image = read_image(user_info[0][2])
            known_encoding = face_recognition.face_encodings(known_image)[0]
            face_result = face_info(new_path)

            if len(face_result["known_face"]) > 0 and len(face_result["known_encoding"]) > 0 and face_result["eyes"]:
                if len(face_result["known_face"]) > 1 or len(face_result["known_encoding"]) > 1:
                    response_pickled = jsonpickle.encode({'message': 'Multiple Faces detected. Please try again.'})
                    logging.info("****************END LOGS FOR VERIFY API****************")
                    rest_params = {"mspin":username, "is_verified":0, "message": 'Multiple Faces detected. Please try again.'}
                    requests.post(url="https://ilearnservice.marutisuzuki.com/import_user_data/update_face_verify.php",data=rest_params)
                    return Response(response=response_pickled, status=200, mimetype="application/json")
                else:
                    unknown_encoding = face_result["known_encoding"][0]
                    distance = face_recognition.face_distance(
                        [known_encoding], unknown_encoding)[0]
                    verify = False
                    message = ''
                    rest_params = {}
                    if distance < 0.5:
                        verify = True
                        message = "Face verified successfully!"
                        rest_params = {"mspin":username, "is_verified":1}
                        # response = requests.post(url="https://ilearnservice.marutisuzuki.com/import_user_data/update_face_verify.php",data=rest_params)
                    else:
                        message = "face did not match"
                        rest_params = {"mspin":username, "is_verified":0, "message": message}

                    response = requests.post(url="https://ilearnservice.marutisuzuki.com/import_user_data/update_face_verify.php",data=rest_params)
                    response = {'verified': verify, 'message': message}
                    logging.info("VERIFIED STATE =====>" + str(verify))
                    data = jsonpickle.encode(response)
                    update_result([username, user_info[0][0],
                                new_path, verify, message])
                    logging.info("****************END LOGS FOR VERIFY API****************")
                    return Response(response=data, status=200, mimetype="application/json")
            else:
                error_info = 'No Valid Face found in uploaded image, Please try again'
                response = {'message': error_info}
                data = jsonpickle.encode(response)
                
                logging.warning(
                    'No Face found in uploaded image for MSPIN =>' + username)
                update_result([username, user_info[0][0],
                              new_path, False, error_info])
                rest_params = {"mspin":username, "is_verified":0, "message": error_info}
                        
                response = requests.post(url="https://ilearnservice.marutisuzuki.com/import_user_data/update_face_verify.php",data=rest_params)
                logging.info("****************END LOGS FOR VERIFY API****************")
                return Response(response=data, mimetype="application/json")
        else:
            response = {'message': 'no file found to compare'}
            data = jsonpickle.encode(response)
            logging.info("****************END LOGS FOR VERIFY API****************")
            return Response(response=data, status=404, mimetype="application/json")

    if params_error:
        response = {'message': 'invalid arguments'}
        data = jsonpickle.encode(response)
        logging.info("****************END LOGS FOR VERIFY API****************")
        return Response(response=data, status=400, mimetype="application/json")

# Test route for load testing
@app.route("/test", methods=["POST"])
def Test():

    known_image = read_image('upload/test.jpeg')
    known_encoding = face_recognition.face_encodings(known_image)[0]
    face_result = face_info('upload/test.jpeg')

    if len(face_result["known_face"]) > 0 and len(face_result["known_encoding"]) > 0 and face_result["eyes"]:
        if len(face_result["known_face"]) > 1 or len(face_result["known_encoding"]) > 1:
            response_pickled = jsonpickle.encode({'message': 'Multiple Faces detected. Please try again.'})
            return Response(response=response_pickled, status=200, mimetype="application/json")
        else:
            unknown_encoding = face_recognition.face_encodings(known_image)[0]
            distance = face_recognition.face_distance(
                [known_encoding], unknown_encoding)[0]
            verify = False
            message = ''
            if distance < 0.5:
                verify = True
                message = "Face verified successfully!"
            else:
                message = "face did not match"
            response = {'verified': verify, 'message': message}
            data = jsonpickle.encode(response)
            return Response(response=data, status=200, mimetype="application/json")
    else:
        error_info = 'No Valid Face found in uploaded image, Please try again'
        response = {'message': error_info}
        data = jsonpickle.encode(response)
        return Response(response=data, mimetype="application/json")

@app.route("/result", methods=["get"])
def show_result():
    username = request.args.get('mspin')
    if username:
        data = get_result(username)
        if(data):
            return render_template('index.html', result=data, len=len(data))
        else:
            return render_template_string('''<!doctype html><html><head></head><body><H1>No Records found</H1></body></html>''')
    else:
        return render_template_string('''<!doctype html><html><head></head><body><H1>Please provide mspin</H1></body></html>''')


@app.route("/get_results", methods=["get"])
def get_all_results():
    payload = []
    content = {}
    username = request.args.get('mspin')
    for result in get_results(username):
        verification_status = 'Failed'
        if result[4]:
            verification_status = 'Verified'
        error = None
        if result[5]:
            error = result[5]

        content = {'mspin': result[1], 'result': verification_status,
                   'error': error, 'date': str(result[6])}
        payload.append(content)
        content = {}
    return jsonpickle.encode(payload)
