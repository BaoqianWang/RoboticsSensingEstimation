import numpy as np
import load_data as ld

def filter_laser_scan(laser_scan,angles):
    #Get rid of the scan that is too far or too close
    valid=np.logical_and((laser_scan<30),(laser_scan>0.1))
    laser_scan = laser_scan[valid]
    angles=angles[valid]
    return laser_scan, angles

def generateMap(res,xmin,ymin,xmax,ymax):
    MAP = {}
    MAP['res'] = res
    MAP['xmin'] = xmin
    MAP['ymin'] = ymin
    MAP['xmax'] = xmax
    MAP['ymax'] = ymax
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

    MAP['log_map'] = np.zeros((MAP['sizex'], MAP['sizey']))
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype = np.int8)
    MAP['show_map'] = 0.5 * np.ones((MAP['sizex'], MAP['sizey'], 3), dtype = np.int8)

    return MAP

def pixels2world(robotWorldFrame,head_angles, depthImageP):

    yaw = robotWorldFrame[2]
    xrobot = robotWorldFrame[0]
    yrobot = robotWorldFrame[1]
    pitch = 0
    roll = 0

    r11 = np.cos(yaw) * np.cos(pitch)
    r12 = np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll)
    r13 = np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)

    r21 = np.sin(yaw) * np.cos(pitch)
    r22 = np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll)
    r23 = np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)

    r31 = -np.sin(pitch)
    r32 = np.cos(pitch) * np.sin(roll)
    r33 = np.cos(pitch) * np.cos(roll)

    t_w2b = np.array([[r11, r12, r13, xrobot],
                      [r21, r22, r23, yrobot],
                      [r31, r32, r33, 0.93],
                      [0, 0, 0, 1]])

    t_b2h = np.array([[np.cos(head_angles[0]), -np.sin(head_angles[0]), 0, 0],
                      [np.sin(head_angles[0]), np.cos(head_angles[0]), 0, 0],
                      [0, 0, 1, 0.33],
                      [0, 0, 0, 1]])

    t_h2l = np.array([[np.cos(head_angles[1]), 0, np.sin(head_angles[1]), 0],
                      [0, 1, 0, 0],
                      [-np.sin(head_angles[1]), 0, np.cos(head_angles[1]), 0.07],
                      [0, 0, 0, 1]])

    t_w2h = np.matmul(t_w2b, t_b2h)
    t_w2l = np.matmul(t_w2h, t_h2l)
    depthImage2D=np.reshape(depthImageP,(depthImageP.shape[0]*depthImageP.shape[1],4))

    imageWorld = np.matmul(t_w2l, depthImage2D.transpose())

    return imageWorld.transpose()


def depthImage2depthFrame(depthImage):
    IRcalibration=ld.getIRCalib()
    intrinsicMatrix=np.array([[IRcalibration['fc'][0],0,IRcalibration['cc'][0]],[0,IRcalibration['fc'][1],IRcalibration['cc'][1]],[0,0,1]])
    inverseIM=np.linalg.inv(intrinsicMatrix)

    depthFrameP=np.ones((depthImage.shape[0],depthImage.shape[1],4))

    for i in range(depthImage.shape[0]):
        for j in range(depthImage.shape[1]):
            if(depthImage[i,j]!=0):
                depthFrameP[i,j,:3]=np.matmul(inverseIM,np.array([[0.01*depthImage[i,j]*(i+1)],[0.01*depthImage[i,j]*(j+1)],[0.01*depthImage[i,j]]])).reshape((3,))
    valid=np.logical_and((depthFrameP[:,:,0]!=1),(depthFrameP[:,:,1]!=1),(depthFrameP[:,:,2]!=1))
    realDepthFrame=depthFrameP[valid,:]
    return realDepthFrame

def depthFrame2imageFrame(depthImageP,rgbImage):
    RGBcalibration = ld.getRGBCalib()
    extrinsicsIR_RGB=ld.getExtrinsics_IR_RGB()
    intrinsicsRGB= np.array([[RGBcalibration['fc'][0],0,RGBcalibration['cc'][0]],[0,RGBcalibration['fc'][1],RGBcalibration['cc'][1]],[0,0,1]])


    transformMatrix=np.concatenate((extrinsicsIR_RGB['rgb_R_ir'],extrinsicsIR_RGB['rgb_T_ir'].reshape((3,1))),axis=1)
    transformMatrix=np.concatenate((transformMatrix,np.array([0,0,0,1]).reshape((1,4))),axis=0)
    pixelsImage=np.zeros((depthImageP.shape[0]*depthImageP.shape[1],3))
    #imageFrameP = np.zeros((depthImageP.shape[0],depthImageP.shape[1], 3))
    halfTransfrom=np.matmul(intrinsicsRGB,np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
    totalTransform=np.matmul(halfTransfrom,transformMatrix)

    imageFrameP=np.matmul(totalTransform,depthImageP.T).T
    for i in range(imageFrameP.shape[0]):
        #There is a problem
        #for j in range(imageFrameP.shape[1]):
            #imageFrameP[i,j,:]=np.matmul(totalTransform,depthImageP[i,j,:])
            u=np.ceil(imageFrameP[i,0]/imageFrameP[i,2]).astype(int)
            v=np.ceil(imageFrameP[i,1]/imageFrameP[i,2]).astype(int)
            pixelsImage[i,:]=rgbImage[u,v,:]
    pixelsImage=np.reshape(pixelsImage,(pixelsImage.shape[0]*pixelsImage.shape[1],3))
    return pixelsImage




def lidar2world(robotWorldFrame, head_angles, lidarData):
    yaw = robotWorldFrame[2]
    xrobot = robotWorldFrame[0]
    yrobot = robotWorldFrame[1]
    pitch = 0
    roll = 0
    r11 = np.cos(yaw) * np.cos(pitch)
    r12 = np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll)
    r13 = np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)

    r21 = np.sin(yaw) * np.cos(pitch)
    r22 = np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll)
    r23 = np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)

    r31 = -np.sin(pitch)
    r32 = np.cos(pitch) * np.sin(roll)
    r33 = np.cos(pitch) * np.cos(roll)

    t_w2b = np.array([[r11, r12, r13, xrobot],
                      [r21, r22, r23, yrobot],
                      [r31, r32, r33, 0.93],
                      [0, 0, 0, 1]])

    t_b2h = np.array([[np.cos(head_angles[0]), -np.sin(head_angles[0]), 0, 0],
                      [np.sin(head_angles[0]), np.cos(head_angles[0]), 0, 0],
                      [0, 0, 1, 0.33],
                      [0, 0, 0, 1]])

    t_h2l = np.array([[np.cos(head_angles[1]), 0, np.sin(head_angles[1]), 0],
                      [0, 1, 0, 0],
                      [-np.sin(head_angles[1]), 0, np.cos(head_angles[1]), 0.15],
                      [0, 0, 0, 1]])

    t_w2h = np.matmul(t_w2b, t_b2h)
    t_w2l = np.matmul(t_w2h, t_h2l)

    lidarDataWorld = np.matmul(t_w2l, lidarData.transpose())

    return lidarDataWorld.transpose()
    

def getPose(lidar):
    time_length=len(lidar)
    pose=np.zeros((3,time_length)) #The last pose is not used
    initial_pose=lidar[0]['delta_pose']
    initial_pose=initial_pose.reshape((3))
    pose[:,0]=initial_pose
    for i in range(time_length-1):
        delta_pose=lidar[i]['delta_pose']
        delta_pose=delta_pose.reshape((3))
        pose[:,i+1]=delta_pose
    return pose
