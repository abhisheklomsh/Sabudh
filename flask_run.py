from flask import Flask

app = Flask(__name__)

@app.route('/')

def func_name():
    return "hello"

if __name__ == "__main__":
    app.run()