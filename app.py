from flask import Flask, render_template, request
from test import main_1
app = Flask(__name__)

@app.route('/search-engine',methods=['GET','POST'])
def index():
	if request.method == 'POST':
		query = request.form['query']
		result = main_1(query)
		return render_template('index.html',data={'query':query,'result':result})
	return render_template('index.html')

if __name__ == '__main__':
   app.run(host='0.0.0.0',debug = True)