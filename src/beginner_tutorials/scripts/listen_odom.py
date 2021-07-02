#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry as odom



def sync_odom(data):
    pub_odom = rospy.Publisher('/updated_odom_max_view', odom, queue_size=100)
    #rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data)
    data.header.stamp=rospy.Time.now()
    
    pub_odom.publish(data)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener_odom', anonymous=True)
    rospy.Subscriber('/odom', odom, sync_odom)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()