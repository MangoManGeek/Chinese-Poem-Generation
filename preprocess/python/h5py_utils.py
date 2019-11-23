import numpy as np
import h5py
import os
import sys
from copy import deepcopy


#handle .(period) and slash specially since it is part of path
#replace with \period or \slash-forward when store, recover later
#not using '\forward-slash'  is because \f is a special character
PERIOD='\period'
SLASH='\slash-forward'

'''
This function will save a python dictionary with format {'key':vector} to hdf5 format

target_dict is the target dictionary
f_name is the HDF5 file name including path to it 
mode is file open mode, 'a' for append is the default setting

output file is a HDF5 file which the keys of HDF5 are keys of target dict
dataset is corresponding array of each key in target dict
'''
def dict2hdf5(target_dict, f_name, sub_group='/', mode='a'):

	print ('Saving In HDF5 Format...')
	index_count=0
	bar_count=0
	total=len(target_dict)
	if total<50:
		total=50

	#open HDF5 file
	f = h5py.File(f_name, mode)

	for key in target_dict:

		######### print progress #############
		index_count +=1
		if index_count % (total/50) ==0:
			sys.stdout.write('\r'),
			bar_count += 1
			sys.stdout.write("[%-50s] %d%% %d/%d" % ('#'*bar_count, 2*bar_count, index_count, total))
			sys.stdout.flush()
		######### end print progress #########


		#print os.path.join(sub_group,key)
		#sys.stdout.flush()


		#replace special handled chars, will not change string if / is not in it
		dataset_key=key.replace('.',PERIOD).replace('/',SLASH)

		f.create_dataset(os.path.join(sub_group,dataset_key), data=target_dict[key])

	f.close()

'''
This function will convert saved hdf5 file from previous function back to {'key':vector} dictionary
f_name is the HDF5 file name including path to it 
'''
def hdf52dict(f_name, sub_group='/' ):

	rv=dict()

	#open HDF5 file
	f = h5py.File(f_name, 'r')

	group=f[sub_group]

	for key in group:

		#replace back special handled chars
		key=key.replace(PERIOD, '.').replace(SLASH,'/')

		rv[key]=np.array(group[key])

	f.close()

	return rv


def main():

	d=dict()

	d['hello']=[1,0,0,0,0,0,0,0,0,0,0]
	d['world']=[0,1,0,0,0,0,0,0,0,0,0]

	dict2hdf5(target_dict=d, f_name='test')

	rv=hdf52dict('test')
	print rv


if __name__ == '__main__':
	main()











