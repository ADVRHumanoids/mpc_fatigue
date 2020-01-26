#!/usr/bin/env python
# license removed for brevity
import rospy
from geometry_msgs.msg import WrenchStamped

def talker():
    RightWrench = WrenchStamped()
    RightWrench.header.frame_id = "mass2_ee"
    RightWrench.wrench.force.x = 0.4
    RightWrench.wrench.torque.y = 0.3    
    
    
    
    LeftWrench = WrenchStamped()
    LeftWrench.header.frame_id = "mass1_ee"
    LeftWrench.wrench.force.x = 0.4
    LeftWrench.wrench.torque.y = 0.3
    

    pub = rospy.Publisher('WrenchStampedTopic', WrenchStamped, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(100) # 10hz
    
    
    while not rospy.is_shutdown():
        rospy.get_time()
        #rospy.loginfo(hello_str)
        RightWrench.header.stamp = rospy.Time.now()
        LeftWrench.header.stamp = rospy.Time.now()
        pub.publish(RightWrench)
        rate.sleep()
        pub.publish(LeftWrench)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass