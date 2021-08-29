from __future__ import absolute_import, division, unicode_literals
from flask import Flask, render_template, request, redirect, jsonify
import mysql.connector as MySQL

import json
from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

import yaml

app = Flask(__name__)
app.static_folder = 'static'









# Configure db
db = yaml.safe_load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']





authenticator = IAMAuthenticator('12L2Ft6kYfHv2nbrHWPKe8NUqe6UtZti661B3naIclJL')
language_translator = LanguageTranslatorV3(
    version='2018-05-01',
    authenticator=authenticator
)

language_translator.set_service_url('https://api.au-syd.language-translator.watson.cloud.ibm.com/instances/efb3d519-3d92-4ba0-8151-d88a8e483587')


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')
    
@app.route('/annotate', methods=['GET', 'POST'])
def annotate():
    return render_template('annotate.html')
    


def translate(ar_sentence):
    translation = language_translator.translate(
    text=ar_sentence,
    model_id='ar-en').get_result()
    english_translations=translation
    return english_translations["translations"][0]["translation"]
    


@app.route('/annotate_api', methods=['GET', 'POST'])
def annotate_api():
    if request.method == 'POST':
        try:
          # Fetch data from db 
            iraqi_sentence_id,iraqi_sentence=get_sentence()
            remaining,done=get_stats()
            translated_sentence=translate(iraqi_sentence)
        except:
          return jsonify({ 'error':True, 'message':'error catched'})
        
            
        return jsonify({ 'error':False,'iraqi_sentence_id':iraqi_sentence_id,'iraqi_sentence':iraqi_sentence,'done':done,'remaining':remaining,'translated_sentence':translated_sentence})
    return jsonify({ 'error':True, 'message':'wrong request'});
    
    
@app.route('/save_user_rating_api', methods=['GET', 'POST'])
def save_user_rating_api():
    if request.method == 'POST':

        try:
            formData = request.form
            user_rating = formData['smiley']
            iraqi_sentence_id = formData['iraqi_sentence_id']
            mysql = MySQL.connect(host=db['mysql_host'],database=db['mysql_db'],user=db['mysql_user'],password=db['mysql_password'])
            mycursor = mysql.cursor(dictionary=True)
            query=("UPDATE `iraqi_sentences` SET `label`=%(label)s WHERE `id`=%(id)s")
    

            data = {
                'label':user_rating ,
                'id':iraqi_sentence_id
            }
            #print(query,data)
            mycursor.execute(query,data)
            mysql.commit()
        except Exception as e: 
          print(e)
          return jsonify({ 'error':True, 'message':'error catched'})
        
            
        return jsonify({ 'error':False,'message':'saved'})
    return jsonify({ 'error':True, 'message':'wrong request'});


def get_stats():
    column="done"


    select_sentence=("SELECT COUNT(*) AS `done`,(6973 - count(*)) AS `remaining`  FROM `iraqi_sentences` WHERE `label`>0")

    mysql = MySQL.connect(host=db['mysql_host'],database=db['mysql_db'],user=db['mysql_user'],password=db['mysql_password'])
    mycursor = mysql.cursor(dictionary=True)

    mycursor.execute(select_sentence)
    iraqiSentence = mycursor.fetchone()
    mycursor.close()
    mysql.close()

    #check if the value in the database
    if(iraqiSentence == None or (iraqiSentence != None and iraqiSentence[column] == None)):
        #the value not in the database, therefore, get it manually
        return ""
    #else if it is in the database, then take it
    else:
        return iraqiSentence["remaining"],iraqiSentence[column]


def get_sentence():
    column="sentence"


    select_sentence=("SELECT `id`,`sentence` FROM `iraqi_sentences` WHERE `label`=0 LIMIT 1")

    mysql = MySQL.connect(host=db['mysql_host'],database=db['mysql_db'],user=db['mysql_user'],password=db['mysql_password'])
    mycursor = mysql.cursor(dictionary=True)

    mycursor.execute(select_sentence)
    iraqiSentence = mycursor.fetchone()
    mycursor.close()
    mysql.close()

    #check if the value in the database
    if(iraqiSentence == None or (iraqiSentence != None and iraqiSentence[column] == None)):
        #the value not in the database, therefore, get it manually
        return ""
    #else if it is in the database, then take it
    else:
        return iraqiSentence["id"],iraqiSentence[column]


if __name__ == '__main__':

    app.run(debug=True)
