from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from flask import send_from_directory
import os
import io
import zipfile
import pandas as pd
import re 
import csv
from sentence_transformers import SentenceTransformer
from model import *

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))   #Specifying path of the project

UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static')        #path to upload and download files 

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
           
            regex(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename)    #calling regex method            
    return render_template('home.html')


def regex(path, filename):
    with open(path, "r", encoding = 'utf-8',) as infile, open(app.config['UPLOAD_FOLDER'] + '/processed.csv', 'w') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
    
        for row in reader:
            newrow = [re.sub(r'\'[0-9]*\', ', "", str(row))]
            newrow = [re.sub(r'[^0-9a-zA-Z.,? ]+', "", str(newrow))]
            
            writer.writerow(newrow)
    outfile.close()
    with open(app.config['UPLOAD_FOLDER'] + '/processed.csv') as inp, open(app.config['UPLOAD_FOLDER'] + '/two_processed.csv', 'w', newline = '') as out:
        writer = csv.writer(out)
        for row in csv.reader(inp):
            
            if any(field.strip() for field in row):
                writer.writerow(row)
   
    c = Clustering(app.config['UPLOAD_FOLDER'] )        #creating an object for clustering class

    c.elbow(app.config['UPLOAD_FOLDER'] )               #Calling the elbow method
   

@app.route('/run', methods=['POST'])
def my_form_post(): 
    c1 = Clustering(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':
        
        k = request.form['k_value']
       
        c1.cluster(app.config['UPLOAD_FOLDER'], int(k))   #calling cluster method
        

    return render_template('run.html')

@app.route('/run/clusters', methods=['POST'])
def zip_cluster():
    if request.method == 'POST':
        if os.path.isdir(app.config['UPLOAD_FOLDER']):      #making zip file for the obtained clusters
                tweets_file = io.BytesIO()
                with zipfile.ZipFile(tweets_file, 'w', zipfile.ZIP_DEFLATED) as my_zip:
                    for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
                        for file in files:
                            if re.search(r'tweets[0-9]', file):
                                my_zip.write(os.path.join(root, file))
                tweets_file.seek(0)
    return send_file(tweets_file, mimetype='application/zip', as_attachment=True, attachment_filename="clusters.zip")



@app.route('/run/summaries', methods=['POST'])
def zip_summary():
    if request.method == 'POST':
        if os.path.isdir(app.config['UPLOAD_FOLDER']):      #making zip file for the obtained summaries
                tweets_file = io.BytesIO()
                with zipfile.ZipFile(tweets_file, 'w', zipfile.ZIP_DEFLATED) as my_zip:
                    for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
                        for file in files:
                            if re.search(r'summary[0-9]', file):
                                my_zip.write(os.path.join(root, file))
                tweets_file.seek(0)
    return send_file(tweets_file, mimetype='application/zip', as_attachment=True, attachment_filename="summaries.zip")


    
    

if __name__ == "__main__":
    app.run(debug=True)


