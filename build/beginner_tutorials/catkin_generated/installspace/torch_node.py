#!/usr/bin/env python2

import torch
import numpy as np
import rospy
import model
import sklearn
from sensor_msgs.msg import LaserScan as laser
from nav_msgs.msg import Odometry as odom
from geometry_msgs.msg import Twist, Vector3


def load_model_txt(model, path):
    data_dict = {}
    fin = open(path, 'r')
    i = 0
    odd = 1
    prev_key = None
    while True:
        s = fin.readline().strip()
        if not s:
            break
        if odd:
            prev_key = s
        else:
            print('Iter', i)
            val = eval(s)
            if type(val) != type([]):
                data_dict[prev_key] = torch.FloatTensor([eval(s)])[0]
            else:
                data_dict[prev_key] = torch.FloatTensor(eval(s))
            i += 1
        odd = (odd + 1) % 2

    # Replace existing values with loaded

    print('Loading...')
    own_state = model.state_dict()
    print('Items:', len(own_state.items()))
    for k, v in data_dict.items():
        if not k in own_state:
            print('Parameter', k, 'not found in own_state!!!')
        else:
            try:
                own_state[k].copy_(v)
            except:
                print('Key:', k)
                print('Old:', own_state[k])
                print('New:', v)
                #sys.exit(0)
    #print('Model loaded')
    return model
    
def get_model(path):
    my_model = model.model([200,1000,500,100,20], "ReLu", 0)
    my_model = load_model_txt(my_model,path)
    return my_model
    
def inference(data, args):
    pub = rospy.Publisher('/ML_odom', odom, queue_size=100)
    model = args[0]
    preprocessing_params = args[1]
    input_max, input_min, output_max, output_min = preprocessing_params
    input_tensor = torch.from_numpy(np.asarray(data.ranges))
    scaled_input = (input_tensor - input_min) / (input_max - input_min)
    
    model.eval()
    with torch.no_grad():
        velocity_prediction = model(scaled_input) 
        print(type(velocity_prediction), type(output_max), type(output_min)) 
        velocity_prediction = velocity_prediction*(output_max-output_min) + output_min
        linear_velocity = velocity_prediction[0].item()
        angular_velocity = velocity_prediction[0].item()
    command_vel = Twist()
    command_vel.linear = Vector3(linear_velocity,0.0,0.0)
    command_vel.angular = Vector3(0.0, 0.0, angular_velocity)
    pub.publish(command_vel) 
   

def predictor(model, preprocessing_params):
    rospy.init_node('ML_teleop', anonymous=True)
    rospy.Subscriber('/updated_laser_max_view', laser, inference, (model, preprocessing_params))
    rate = rospy.Rate(10)
    rospy.spin()
if __name__ == '__main__':
    model_path = str("/home/sandip.patel/docs_for_ml_teleops/ML/model.txt")
    model = get_model(path=model_path)
    input_max = torch.from_numpy(np.load("/home/sandip.patel/docs_for_ml_teleops/ML/input_max.npy"))
    input_min = torch.from_numpy(np.load("/home/sandip.patel/docs_for_ml_teleops/ML/input_min.npy"))
    output_max = torch.from_numpy(np.load("/home/sandip.patel/docs_for_ml_teleops/ML/output_max.npy"))
    output_min = torch.from_numpy(np.load("/home/sandip.patel/docs_for_ml_teleops/ML/output_min.npy"))
    preprocessing_params = (input_max, input_min, output_max, output_min)
    predictor(model, preprocessing_params)