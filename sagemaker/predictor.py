# -*- coding: utf-8 -*-

import os
import json
import boto3
from dalle_model import DalleModel
import jax
import time

import warnings

warnings.filterwarnings("ignore",category=FutureWarning)

import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import flask

# The flask app for serving predictions
app = flask.Flask(__name__)

s3_client = boto3.client('s3')

print ("loading models!")
print("<<< devices", jax.devices())
dalle_model = DalleModel("Mini")
print ("<<<< loading  models success")


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")
    data = flask.request.data.decode('utf-8')
    print("<<<<<input data: ", data)
    
    data = json.loads(data)
    data_input = data['data']
    bucket_name = data['bucket_name']

    # inference
    res = dalle_model.generate_images(data_input, 1)
    
    print("Done inference! ")
    print("res: ", res)

    #send back reasult to s3
    dir_name = '/app'
    for idx, img in enumerate(res):
        img.save(os.path.join(dir_name, f'1.JPEG'), format="JPEG")

    #upload
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file('/app/1.JPEG', bucket_name, '1.JPEG')
    print ("upload succuss!")

    inference_result = {
        'result': "success!"
    }
    
    _payload = json.dumps(inference_result, ensure_ascii=False)
    return flask.Response(response=_payload, status=200, mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
