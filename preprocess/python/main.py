import os

import json_utils
import process_raw




JSON_DATA_PATH='../../../data/json/'
HDF5_DATA_PATH='../../../data/hdf5/'

def main():

	##### process all poem.tang json files and save as hdf5 #####


	#get a list of files to process
	file_names=os.listdir(JSON_DATA_PATH)

	for f in file_names:

		if not (f.startswith('poet.tang') and f.endswith('.json')):
			# ignore all poems that are not tang poem
			continue

		raw_data=json_utils.parse_json(JSON_DATA_PATH+f)

		poem_dicts=[]
		poem_dicts=process_raw.process_poems(raw_data, poem_dicts)


		hdf5_path=HDF5_DATA_PATH+f[:-5]+'.hdf5'
		process_raw.write_poems_as_hdf5(poem_dicts, hdf5_path)


if __name__ == '__main__':
	main()


