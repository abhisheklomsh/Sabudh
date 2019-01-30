import pandas as pd
import json
from flask import Flask,request,redirect,url_for,jsonify
app=Flask(__name__)

@app.route("/",methods=['GET']) # 'http::/www.google.com/'
def hom():
    df=pd.read_csv("out.csv")
    new_dict=df.to_dict("index")
    abc= list(new_dict.values())
    return json.dumps(abc)

@app.route('/',methods=["POST"])
def insert():
    data=request.get_json()
    df=pd.read_csv('out.csv')
    list_of_df=list(df.to_dict('index').values())
    created_data={}
    created_data['index'] = (list_of_df[-1]["index"]+1)
    combined_data={**created_data,**data}
    df=df.append(combined_data,ignore_index=True)
    print(df)
    df.to_csv(r'out.csv',index=None,header=True)
    return jsonify({"index":created_data["index"]})

@app.route('/',methods=["PUT"])
def update():
    data=request.get_json()
    df=pd.read_csv("out.csv")
    df.loc[df["index"] == data["index"], ['name','roll',"class"]] = data['name'],data['roll'],data["class"]
    df.to_csv(r'out.csv',index=None,header=True)
    return jsonify({"index":data["index"]})

@app.route('/',methods=["DELETE"])
def delete():
    data=request.get_json()
    df=pd.read_csv("out.csv")
    df= df[df["index"]!= data['index']]
    df.to_csv(r'out.csv',index=None,header=True)
    return jsonify({"index":data["index"]})

app.run(port=1000)