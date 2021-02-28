import numpy as np
from flask import Flask, request, jsonify
import pickle

filename = ‘Similarity_model’
load_model = pickle.load(open(filename, 'rb'))
app = Flask(__name__)
@app.route('/api',methods=['POST'])

def predict()
data = request.get_json(force=True)
predict_request=[[imagenet]]
request=np.array(predict_request)
print(request)
prediction = load_model.predict(predict_request)
pred = prediction[0]
print(pred)
return jsonify(int(pred))

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    app.run(port=9000, debug=True)