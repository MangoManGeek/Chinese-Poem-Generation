import json

def parse_json(file_path):
	data={}

	with open(file_path) as f:
		data=json.load(f)

	return data #python dict