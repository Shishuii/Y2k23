from flask import Flask, request, jsonify
import os
import json
import pandas as pd

from inference import Event_qa

app = Flask(__name__)
app.debug = True

e_qa = Event_qa()

@app.route('/', methods=['GET'])
def server_status():
    return jsonify({"status": "OK"})

@app.route('/event_qa', methods=['POST'])
def query_event_data():
    e_id = json.loads(request.data, strict=False)['event_id']
    question = json.loads(request.data, strict=False)['q']
    response1 = e_qa.query(e_id, question)
    final_response = {}
    final_response['query']=question
    final_response['event_id']=e_id
    final_response['answer']=response1
    return jsonify(final_response)

if __name__ == "__main__":
    print("***************************************************")
    print("Running APP")
    app.run(host="0.0.0.0", port=3000, use_reloader=False)
