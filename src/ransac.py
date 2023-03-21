#! /usr/bin/env python
'''
Author : swooeun
github.com/Seungw0o
'''
import rospy
import numpy as np
import ros_numpy
import time
from sensor_msgs.msg import PointCloud2

class LiDARProcess:
    def __init__(self):
        #subscriber
        rospy.Subscriber("velodyne_points", PointCloud2, self.callback)

        #publisher
        self.inlier_pub = rospy.Publisher("/velodyne/inlier", PointCloud2, queue_size=1)
        self.outlier_pub = rospy.Publisher("/velodyne/outlier", PointCloud2, queue_size=1)

    def callback(self, msg):
        start_time = time.time()
        xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, False)
        inlier, outlier = self.ransac_algorithm(xyz_array, 20, 0.3)
    
        # visualization
        inlier_msg = self.convert_pc2_msg(inlier, msg.header)
        outlier_msg = self.convert_pc2_msg(outlier, msg.header)
        self.inlier_pub.publish(inlier_msg)
        self.outlier_pub.publish(outlier_msg)
        print("processing time : {}s". format(time.time()-start_time))

    def convert_pc2_msg(self, cloud_msg, header):
        pc_array = np.zeros(len(cloud_msg), dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('intensity', np.float32),
        ])
        pc_array['x'] = cloud_msg[:, 0]
        pc_array['y'] = cloud_msg[:, 1]
        pc_array['z'] = cloud_msg[:, 2]
        pc_array['intensity'] = 255
        pc_msg = ros_numpy.msgify(PointCloud2, pc_array, stamp=header.stamp, frame_id=header.frame_id)

        return pc_msg


    def ransac_algorithm(self, xyz_array, max_iter, min_dis):
        iter = 0
        len_buff = 0
        inliers_result = []
        outliers_result = []
        while (iter < max_iter):
            rand_idx = np.random.randint(0, len(xyz_array)-1, size=2)
            x1, _, y1 = xyz_array[rand_idx[0]]
            x2, _, y2 = xyz_array[rand_idx[1]]
            a, b, c = y2-y1, -(x2-x1), -(y2-y1)*x1+(x2-x1)*y1

            condition = abs(a*xyz_array[:, 0]+b*xyz_array[:, 2]+c)/np.sqrt(a*a+b*b) < min_dis
            
            inliers = xyz_array[condition==True]
            outliers = xyz_array[condition==False]
            if len_buff < len(inliers):
                inliers_result, outliers_result = inliers, outliers
                len_buff = len(inliers)
            iter += 1

        return inliers_result, outliers_result

if __name__=="__main__":
    rospy.init_node("lidar_processing")
    lp = LiDARProcess()
    rospy.spin()