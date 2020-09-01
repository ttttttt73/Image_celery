# -*- coding: utf-8 -*- 
import os
from flask import Flask, jsonify, request, flash, redirect, url_for, render_template, send_from_directory, abort
from werkzeug.utils import secure_filename
from flask_cors import CORS
import secrets
from celery import Celery
# from tasks import upload_async_photo, get_image, to_gray, to_threshold, find_contour, test_contour
import base64
import cv2
import numpy as np
import io
import ipfshttpclient
import redis
from contour_json import save_json, to_json, compare_point


r = redis.Redis(host='localhost', port=6379, db=0)
api = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')


UPLOAD_FOLDER = './saved'
ALLOWED_EXTENSIONS = {'bmp', 'tiff', 'tif', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def make_celery(app):
    celery = Celery('test_flask', broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    TaskBase = celery.Task
    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask
    return celery


app = Flask(__name__,
            static_folder = "./dist/static",
            template_folder = "./dist")

app.config.update(
    CELERY_BROKER_URL = 'redis://localhost:6379',
    CELERY_RESULT_BACKEND = 'redis://localhost:6379'
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
celery = make_celery(app)
CORS(app)

secret = secrets.token_urlsafe(32)
app.secret_key = secret


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@celery.task()
def imageUpload(filename):
    with open((os.path.join(app.config['UPLOAD_FOLDER'], filename)), 'rb+') as data:
        res = api.add(data)
    r.set(res['Hash'], filename)
    return res


@celery.task()
def get_ipfs(hash_val):
    # url = 'http://localhost:8080/ipfs/'+ hash_val
    # r = requests.get(url)
    rfile = api.cat(hash_val)
    # return r.text
    # return rfile
    # return render_template("image.html", **locals())
    

@celery.task()
def getimage(hash_val):
    path_bytes = r.get(hash_val)
    path = path_bytes.decode('utf-8')
    # path = os.path.join(app.config['UPLOAD_FOLDER'], hash_val)+'.jpg'
    return path


@celery.task()
def grayscale(hash_val):
    # img = get_ipfs.delay(hash_val)
    # img = cv2.imread(filename, cv2.IMREAD_COLOR)
    # img_bytes = base64.b64decode(content)
    # jpg_as_np = np.frombuffer(img_bytes, dtype=np.uint8)
    # img = cv2.imdecode(jpg_as_np, flags=1)
    img_bytes = api.cat(hash_val)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)
    print(img.shape)
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filename_bytes = r.get(hash_val)
    filename = filename_bytes.decode('utf-8')
    root, ext = os.path.splitext(filename)
    save_name = str(root)+'_gray.jpg'
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], save_name), dst)
    ret, buf = cv2.imencode('.jpg', dst)
    content = buf.tobytes()
    res = api.add_bytes(content)
    r.set(res, save_name)
    return res
    # task = imageUpload.delay(save_name)
    # task_result = task.get()
    # return task_result


@celery.task()
def do_threshold(hash_val):
    img_bytes = api.cat(hash_val)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
    ret, dst = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    filename_bytes = r.get(hash_val)
    filename = filename_bytes.decode('utf-8')
    root, ext = os.path.splitext(filename)
    save_name = str(root) + '_thresh.jpg'
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], save_name), dst)
    ret, buf = cv2.imencode('.jpg', dst)
    content = buf.tobytes()
    res = api.add_bytes(content)
    r.set(res, save_name)
    return res


@celery.task()
def find_contour(hash_val, d, e):
    img_bytes = api.cat(hash_val)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
    filename_bytes = r.get(hash_val)
    filename = filename_bytes.decode('utf-8')
    root, ext = os.path.splitext(filename)
    save_name = str(root) + '_contour'
    output_folder = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    index = 0   
    bndbox = []
    thresh_color = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    img = cv2.dilate(img, None, iterations=int(d))
    img = cv2.erode(img, None, iterations=int(e))
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cropped = img[y:y+h, x:x+w]
        output_name = '_' + str(index) + '.jpg'
        h, w = cropped.shape[:2]
        area = w*h
        if 1600 < area < 90000:
            rect = cv2.rectangle(thresh_color, (x,y), (x+w,y+h), (0,255,0), 2)
            index = index + 1
            cv2.imwrite(os.path.join(output_folder, save_name+output_name), cropped)
            bndbox.append([index, x, y, w, h])
    result = '.jpg'
    cv2.imwrite(os.path.join(output_folder, save_name+result), thresh_color)
    js = to_json(save_name+result, bndbox)
    save_json(js, os.path.join(output_folder, save_name)+'.json')
    ret, buf = cv2.imencode('.jpg', thresh_color)
    content = buf.tobytes()
    res = api.add_bytes(content)
    r.set(res, os.path.join(save_name, save_name+result))
    return res
    

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/img/<path:filename>')
def send_file(filename):
    print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/getimage')
def get_image():
    temp = request.args.get('hash', '')
    task = getimage.delay(temp)
    task_result = task.get()
    # return task_result
    return send_from_directory(app.config['UPLOAD_FOLDER'], task_result)


@app.route('/imageUpload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
            # flash('No file part')
            # return redirect(request.url)
        f = request.files['file']
        # file 없을 때 예외처리
        if f.filename == '':
            return '파일이 없습니다.'
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)  
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            task = imageUpload.delay(filename)
            task_result = task.get()
            res = {'hash': task_result}
            return task_result
            # return jsonify(res)


@app.route('/grayscale')
def gray_scale():
    temp = request.args.get('hash', '')
    # task = get_ipfs.delay(temp)
    # task_result = task.wait()
    task = grayscale.delay(temp)
    task_result = task.wait()
    res = {'hash': task_result}
    return jsonify(res)


@app.route('/threshold')
def threshold():
    temp = request.args.get('hash', '')
    task = do_threshold.delay(temp)
    task_result = task.wait()
    res = {'hash': task_result}
    return jsonify(res)


@app.route('/contour')
def contour():
    temp = request.args.get('hash', '')
    d = request.args.get('d', 0)
    e = request.args.get('e', 0)
    task = find_contour.delay(temp, d, e)
    task_result = task.wait()
    res = {'hash': task_result}
    return jsonify(res)


@app.route('/bndbox')
def bndbox():
    temp = request.args.get('hash', '')
    task = getimage.delay(temp)
    task_result = task.wait()
    point_x = request.args.get('x', -1)
    point_y = request.args.get('y', -1)
    print('(x, y): ', point_x, point_y)
    root, ext = os.path.splitext(task_result)
    print('root: ', root)
    selected = compare_point(os.path.join(app.config['UPLOAD_FOLDER'], root+'.json'), point_x, point_y)
    print(selected)
    if selected is None:
        abort(404)
    else:
        filename = selected['filename']
        index = selected['index']
        root, ext = os.path.splitext(filename)
        sel_filepath = []
        for i in range(len(index)):
            sel_filepath.append(os.path.join('/'+root+'/'+root+'_'+str(index[i])+ext))
        print(sel_filepath)
        return jsonify(sel_filepath) 


@app.route('/vision')
def vision():
    temp = request.args.get('hash', '')
    task = getimage.delay(temp)
    task_result = task.wait()
    pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
