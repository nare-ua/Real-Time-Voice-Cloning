from flask import (
  Flask, 
  Response, 
  flash,
  render_template, 
  request, 
  session,
  redirect,
  url_for,
  send_from_directory
)

import rtvc

from werkzeug.utils import secure_filename

import sys
# Tornado web server
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
import tornado.ioloop
from tornado.ioloop import IOLoop
#from text2speech import T2S
import os

#language = 'kr' 
#t2s = T2S(language)
#sample_text = {
#    'kr' : '여기에 텍스트 입력',
#    'en' : 'Enter the text'
#}

# Initialize Flask.


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

app.debug = True

app.config['SESSION_TYPE'] = 'filesystem'

UPLOAD_FOLDER = '/mnt/data/uploads'
ALLOWED_EXTENSIONS = {'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#@app.route('/tts', methods=['GET', 'POST'])
#def texttospeech():
#    if request.method == 'POST':
#        result = request.form
#        lang = result['input_language']
#        text = result['input_text']
#        if lang == t2s.language:
#            audio = t2s.tts(text)
#        else:
#            audio = t2s.update_model(lang).tts(text)
#        print(audio)
#        return render_template('simple.html', voice=audio, sample_text=text, opt_lang=t2s.language)

            
#Route to render GUI
@app.route('/')
def show_entries():
  #return render_template('simple.html', sample_text=sample_text.get(t2s.language), voice=None, opt_lang=t2s.language)
  return render_template('rec.html')

@app.route('/rec')
def show_rec():
  return render_template('rec.html')

@app.route('/tss/<path:path>')
def static_file(path):
  print("static_file:",path)
  return send_from_directory(app.config['UPLOAD_FOLDER'], path)

@app.route('/upload', methods=['POST'])
def upload_file():
  # check if the post request has the file part
  input_text = request.form.get('text')
  print("input_text=", input_text)
  print("request.files=", request.files)
  print("request.files.keys=", list(request.files.keys()))

  if 'file' not in request.files:
    flash('No file part')
    return redirect(request.url)

  _file = request.files['file']
  print("file=", _file)
  print("file.filename=", _file.filename)
  print(allowed_file(_file.filename))


  # if user does not select file, browser also
  # submit an empty part without filename
  if _file.filename == '':
    flash('No selected file')
    return redirect(request.url)

  if _file and allowed_file(_file.filename):
    filename = secure_filename(_file.filename)
    b, ext = os.path.splitext(filename)
    filename_dn = b + "_dn" + ext
    filename_sr = b + "_sr" + ext
    filename_tss = b + "_tss" + ext
    
    __filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    __filepath_dn = os.path.join(app.config['UPLOAD_FOLDER'], filename_dn)
    __filepath_sr = os.path.join(app.config['UPLOAD_FOLDER'], filename_sr)
    __filepath_tss = os.path.join(app.config['UPLOAD_FOLDER'], filename_tss)
    print("saveing to ...", __filepath)
    _file.save(__filepath)
    print("downsampling to ...", __filepath_dn)
    rtvc.downsample(__filepath, __filepath_dn)
    print("silence removal to ...", __filepath_sr)
    rtvc.silence_removal(__filepath_dn, __filepath_sr)

    if rtvc.convert(input_text, __filepath_sr, __filepath_tss):
        return filename_tss
    else:
        return "FAIL"
    #return redirect(url_for('uploaded_file', filename=filename))

#Route to stream music
#@app.route('/<voice>', methods=['GET'])
#def streammp3(voice):
#    
#    def generate():    
#        with open(os.path.join('wavs',voice), "rb") as fwav:
#            data = fwav.read(1024)
#            while data:
#                yield data
#                data = fwav.read(1024)
#            
#    return Response(generate(), mimetype="audio/mp3")

#launch a Tornado server with HTTPServer.
import os
DEFAULT_FLASK_PORT=10050
if __name__ == "__main__":
  rtvc.init()
  port = os.environ.get('FLASK_PORT', DEFAULT_FLASK_PORT)
  http_server = HTTPServer(WSGIContainer(app))
  http_server.listen(port)
  io_loop = tornado.ioloop.IOLoop.current()
  print("tornado starting...")
  io_loop.start()
    
