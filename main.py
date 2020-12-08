
import sys
import tensorflow as tf
import numpy as np
from keras import layers
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input, Reshape, Add, Dense, Lambda ,Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras import backend
from keras.engine.topology import Layer
import glob
from st_resnet import *
import preprocess_data

def get_training_data(data, sequence_length, trend_gap, period_gap, closeness_gap):
    
    training_data_x = []
    
    training_data_y = []
    di = 0
    ni = 0
    ri = 0
    while((di+(trend_gap*sequence_length)+(period_gap*sequence_length) + (closeness_gap*sequence_length))<len(data)):
#         print("condition: "+str(di+(trend_gap*sequence_length)+(period_gap*sequence_length) + (closeness_gap*sequence_length)))
        distant = None
        near = None
        recent = None
        di_old = di
        for i in range(sequence_length):
#             print("di: ", di)
            if(distant is None):
                distant = data[di]
            else:
                distant = np.concatenate((distant, data[di]), axis = 2)
#             distant.append(data[di])
            di+=trend_gap
            
        ni = di-trend_gap+period_gap
        for i in range(sequence_length):
#             print("ni: ", ni)
            if(near is None):
                near = data[ni]
            else:
                near = np.concatenate((near, data[ni]), axis = 2)
#             near.append(data[ni])
            ni+=period_gap
            
        ri = ni-period_gap + closeness_gap
        for i in range(sequence_length):
#             print("ri: ", ri)
            
            if(recent is None):
                recent = data[di]
            else:
                recent = np.concatenate((recent, data[ri]), axis = 2) 
#             recent.append(data[di])
            ri+=closeness_gap
            
#         distant = np.array(distant)
#         print(distant.shape)
        
        training_point = np.concatenate((distant,near,recent), axis=2)
#         print(training_point.shape)
        shape = training_point.shape
#         training_point = training_point.reshape((shape[1], shape[2], shape[0]*shape[3]))
#         print(training_point.shape)
        training_data_x.append(training_point)
        training_data_y.append(data[ri])
        di = di_old + 1
        
    return np.array(training_data_x), np.array(training_data_y)

def read_input():
    files = []
    for filename in glob.glob("./preprocessing/right_block/start/*.txt"):
        files.append(filename)

    files = sorted(files)

    starts = []
    for filename in files:
        starts.append(np.loadtxt(filename, delimiter=","))

    start_array = np.array(starts)
    start_array = start_array.reshape(start_array.shape+tuple([1]))



    files = []
    for filename in glob.glob("./preprocessing/right_block/end/*.txt"):
        files.append(filename)

    files = sorted(files)

    ends = []
    for filename in files:
        ends.append(np.loadtxt(filename, delimiter=","))

    end_array = np.array(ends)
    end_array = end_array.reshape(end_array.shape+tuple([1]))

    data = np.concatenate((start_array, end_array), axis=3)
    
    return data


def train(data, sequence_length = 4):
    X_t, Y_t = get_training_data(data, sequence_length, 4, 2, 1)
    x_closeness = X_t[:,:,:,:sequence_length*2]
    x_period = X_t[:,:,:,sequence_length*2:2*sequence_length*2]
    x_trend = X_t[:,:,:,2*sequence_length*2:]

    model = st_resnet((32,32, 6*sequence_length), 12, 64, (3,3),np.min(Y_t, axis=0), np.max(Y_t, axis=0) )
    model.compile(optimizer='adam', loss='mean_squared_error', metrics= ['accuracy'])

    model_history = model.fit([x_closeness,x_period,x_trend], Y_t, epochs = 1, batch_size = 32, verbose = 2)



if __name__ == "__main__":
    try:
        choice = sys.argv[1]
        choice = int(choice)
    except:
        print("\n\n\n\n")
        print("Invalid choice! Please select a valid choice")
        exit()

    if choice == 1:
        data = read_input()
        train(data)
    elif choice == 2:
        preprocess_data.main()


