#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan as laser


def sync_laser(data):
    
    #rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data)
    pub_laser = rospy.Publisher('/updated_laser_max_view', laser, queue_size=100)
    data.header.stamp=rospy.Time.now()
    
    pub_laser.publish(data)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener_scan', anonymous=True)
    rospy.Subscriber('/laserscan_from_cloud_max_view', laser, sync_laser)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()