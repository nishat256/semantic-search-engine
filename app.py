from flask import Flask, render_template, request
from process_query import main
app = Flask(__name__)

@app.route('/search-engine',methods=['GET','POST'])
def index():
	if request.method == 'POST':
		query = request.form['query']
		result = main(query)
		return render_template('index.html',data={'query':query,'result':result})
	return render_template('index.html')

if __name__ == '__main__':
   app.run(host='0.0.0.0',debug = True)