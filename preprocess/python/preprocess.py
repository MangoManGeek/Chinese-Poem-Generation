import os
import h5py

import json_utils
import process_raw


JSON_DATA_PATH='../../../data/json/'
HDF5_DATA_PATH='../../../data/hdf5/'


# @return poems: a list of all poem dictionarys
def get_data():

	# get all hdf5 data to a list of dictionarys

	file_names=os.listdir(HDF5_DATA_PATH)

	poems=[]

	for f in file_names:

		hdf5_path=HDF5_DATA_PATH+f
		p=process_raw.read_poems_from_hdf5(hdf5_path)

		poems += p

	return poems

