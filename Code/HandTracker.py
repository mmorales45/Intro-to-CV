'''
This program tracks the user's hand and moves a pincherX 100 robot to mimic the user's hand movement.
'''
import pyrealsense2 as rs
import numpy as np 
import cv2
import math
from interbotix_xs_modules.arm import InterbotixManipulatorXS

#Initialize the robot and have it move ot home pose then open gripper
robot = InterbotixManipulatorXS("px100",'arm','gripper')
robot.arm.go_to_home_pose()
robot.gripper.open()


def change_depth(thetas):
    '''
    Move the robot's depth according to the given theta values
    '''
    robot.arm.set_joint_positions(thetas)


def change_vert(theta):
    '''
    Move the robot's vertical position according to the given theta 
    '''
    # robot.arm.set_single_joint_position("waist", 0)
    robot.arm.set_single_joint_position("shoulder", theta)

def change_side(theta):
    '''
    Move the robot's horizontal position according to the given theta 
    '''
    robot.arm.set_single_joint_position("waist", theta)


def finger_point(defects, contour, centroid):
    '''
    Determine the farthest point on the hand from the centroid of the contour
    '''
    if defects is not None and centroid is not None:
        Num_list = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[Num_list][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[Num_list][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        max_dist = np.argmax(dist)

        if max_dist < len(Num_list):
            defect_point = Num_list[max_dist]
            finger_point = tuple(contour[defect_point][0])
            return finger_point
        else:
            return None

#Enable the real sense pipeline
cfg = rs.pipeline()  #good
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = cfg.start(config) #good

#Get the color video stream and get the intrinsics of the camera
p= profile.get_stream(rs.stream.color)
intrinsics = p.as_video_stream_profile().get_intrinsics()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

#Block anything farther than 0.5 meters
clipping_distance_in_meters = 0.5 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

align_to = rs.stream.color
align = rs.align(align_to)
work = True#HSV FOR RED
#Initalize the states
gripper_open = 0
gripper_close = 0
mode_left = 0
mode_right = 0
mode_down = 0
mode_up = 0
mode_forward = 0
mode_back = 0
mode_center = 0
mode_middle = 0
#Uncomment below to create trackbar
# cv2.namedWindow("HSV Value")
# cv2.createTrackbar("H MIN", "HSV Value", 0, 179, callback)
# cv2.createTrackbar("S MIN", "HSV Value", 0, 255, callback)
# cv2.createTrackbar("V MIN", "HSV Value", 0, 255, callback)
# cv2.createTrackbar("H MAX", "HSV Value", 179, 179, callback)
# cv2.createTrackbar("S MAX", "HSV Value", 255, 255, callback)
# cv2.createTrackbar("V MAX", "HSV Value", 255, 255, callback)
try:
    while work:
        #Get the frames
        frames = cfg.wait_for_frames()

        aligned_frames = align.process(frames)
        #Align the frames for depth and color
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        #Create HSV image
        hsv = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2HSV)
        #Uncomment below to create trackbar
        # h_min = cv2.getTrackbarPos("H MIN", "HSV Value")
        # s_min = cv2.getTrackbarPos("S MIN", "HSV Value")
        # v_min = cv2.getTrackbarPos("V MIN", "HSV Value")
        # h_max = cv2.getTrackbarPos("H MAX", "HSV Value")
        # s_max = cv2.getTrackbarPos("S MAX", "HSV Value")
        # v_max = cv2.getTrackbarPos("V MAX", "HSV Value")
        # lower_skin = np.array([h_min, s_min, v_min])
        # upper_skin = np.array([h_max, s_max, v_max])

        #Skin hsv vlaues
        lower_skin = np.array([1, 131, 43])
        upper_skin = np.array([17, 221, 205])
        #create mask based on hsv values then use dilate and blur
        mask = cv2.inRange(hsv, lower_skin, upper_skin) 
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        blur = cv2.GaussianBlur(mask,(3,3),0)
        res = cv2.bitwise_and(bg_removed,bg_removed, mask=mask)
        
        num_cont = -1
        #Create contours
        contours, hierarchy = cv2.findContours(blur,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #Draw contours
        cv2.drawContours(bg_removed, contours, num_cont, (0,0,255), 3)
 
        #Show the mask,blurred mask, res and the color image
        cv2.imshow('mask',mask)
        cv2.imshow('mask_blur',blur)
        
        cv2.imshow('res',res)
        cv2.imshow('Orignal Image',color_image)
        try:
            #Determine the largest contour in the image and create a moment
            cnt = max(contours,key=cv2.contourArea)
            M = cv2.moments(cnt)
            #Check to see if the contour is significant
            if len(cnt) > 10:
                #Calcualte the centroids c and y value
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                #Draw circles on the centroid
                cv2.circle(bg_removed, (cx,cy), 10, (255,255,255), -1)
                cv2.circle(bg_removed, (cx,cy), 30, (0,255,0), 5)
                #Calculate the x,y,z coordinates of the centroid
                position = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_image[cy][cx]*0.001)
                hull = cv2.convexHull(cnt, returnPoints=False)
                defects = cv2.convexityDefects(cnt, hull)
                #Find the farthest point from the centroid of the hand
                far_point = finger_point(defects, cnt, (cx,cy))
                cv2.circle(bg_removed, (far_point[0],far_point[1]), 10, (255, 255, 255), -1)
                #Calculate the distance between centroid and tip of finger/farthest point on hand
                diff = math.hypot(cx-far_point[0],cy-far_point[1])
                print('difference in length',diff) 

                #draw the line that connects the main centroid and the outlier
                cv2.line(bg_removed, (cx,cy), (far_point[0],far_point[1]), (0,255,0), 3)
                print(position[2])
                #Move arm left
                if position[0] > 0.05:
                    print("Left")
                    theta = -np.pi/2
                    if mode_left == 1:
                        pass
                    else:
                        change_side(theta)
                        mode_left = 1
                        mode_right = 0
                        mode_center = 0
                #Move arm right
                elif position[0] < -0.05:
                    print("Right")
                    theta = np.pi/2
                    if mode_right == 1:
                        pass
                    else:
                        change_side(theta)
                        mode_right = 1
                        mode_left = 0
                        mode_center = 0
                #Move arm to center and change the depth of the camera based on depth value
                elif position[0] >= -0.05 and position[0] <= 0.05:
                    print("Center")
                    theta = 0
                    if mode_center == 1:
                        pass
                    else:
                        change_side(theta)
                        mode_center = 1
                        mode_right = 0
                        mode_left = 0
                    if position[1] >= -0.05 and position[1] <= 0.05:
                        if position[2] > 0.4 and position[2] <0.5:
                            thetas = [0,np.pi/8,-np.pi/8,0]
                            print('passed')
                            if mode_forward == 1:
                                pass
                            else:
                                change_depth(thetas)
                                mode_forward = 1 
                                mode_back = 0
                                mode_norm = 0
                        if position[2] <0.3:
                            thetas = [0,-np.pi/8,np.pi/8,0]
                            print('passed')
                            if mode_back == 1:
                                pass
                            else:
                                change_depth(thetas)
                                mode_back = 1 
                                mode_forward = 0
                                mode_norm = 0
                        elif position[2] > 0.3 and position[2] <0.4:
                            thetas = [0,0,0,0]
                            print('passed')
                            if mode_norm == 1:
                                pass
                            else:
                                change_depth(thetas)
                                mode_norm = 1
                                mode_back = 0
                                mode_forward = 0
                #Move robot down
                if position[1] > 0.05:
                    print("DOWN")
                    theta = np.pi/16
                    if mode_down == 1:
                        pass
                    else:
                        change_vert(theta)
                        mode_down = 1
                        mode_up = 0
                        mode_right = 0
                        mode_left = 0
                        mode_middle = 0
                #Move robot up
                elif position[1] < -0.05:
                    print("UP")
                    theta = -np.pi/16
                    if mode_up == 1:
                        pass
                    else:
                        change_vert(theta)
                        mode_up = 1
                        mode_down = 0
                        mode_right = 0
                        mode_left = 0
                        mode_middle = 0

                #Move robot to center/home vertical distance
                elif position[1] >= -0.05 and position[1] <= 0.05:
                    print("Middle")
                    theta = 0
                    if mode_middle == 1:
                        pass
                    else:
                        change_vert(theta)
                        mode_middle = 1
                        mode_up = 0
                        mode_down = 0
                        mode_right = 0
                        mode_left = 0

                #Close gripper based on depth and line difference, far 
                if position[2] > 0.4 and position[2] <0.5:
                    print('far')
                    print("Forward")

                    if diff < 100:
                        print("close gripper")
                        if gripper_close == 1:
                            pass
                        else:
                            robot.gripper.close()
                            gripper_close == 1
                            gripper_open == 0 
                    elif diff >= 100:
                        print("open gripper")
                        if gripper_open == 1:
                            pass
                        else:
                            robot.gripper.open()
                            gripper_open == 1
                            gripper_close == 0
                #Close gripper based on depth and line difference, medium distance
                elif position[2] > 0.3 and position[2] <0.4:
                    print('medium')

                    if diff < 130:
                        print("close gripper")
                        if gripper_close == 1:
                            pass
                        else:
                            robot.gripper.close()
                            gripper_close == 1
                            gripper_open == 0 
                    elif diff >= 130:
                        print("open gripper")
                        if gripper_open == 1:
                            pass
                        else:
                            robot.gripper.open()
                            gripper_open == 1
                            gripper_close == 0
                #Close gripper based on depth and line difference, near distance
                elif position[2] <0.3:
                    print('near')

                    if diff < 160:
                        print("close gripper")
                        if gripper_close == 1:
                            pass
                        else:
                            robot.gripper.close()
                            gripper_close == 1
                            gripper_open == 0 
                    elif diff >= 160:
                        print("open gripper")
                        if gripper_open == 1:
                            pass
                        else:
                            robot.gripper.open()
                            gripper_open == 1
                            gripper_close == 0


            
        except:
            pass
        
        
        #Display both images, color image and image with the background removed, in a window
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, color_image))


        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    cfg.stop()
cv2.destroyAllWindows()