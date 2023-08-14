from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/hello')
def hello():
   return "Hello !"

@app.route('/classify_image', methods = ['GET','POST'])
def classify_image():
  img_data = request.form['image_data']
  response = jsonify(util.classify_image(img_data))

  response.headers.add('Access-Control-Allow-Origin', '*')

  return response


if(__name__ == "__main__"):
    print("Starting Python Flask server for sports celebrity classification model...")
    util.load_artifacts()
    app.run(port=5000)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         