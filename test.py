import os

def main_1(query):
	os.system('python process_query.py "{}"'.format(query))
	with open('result.txt') as fo:
		content = fo.read()
	if not content:
		return False
	return eval(content)