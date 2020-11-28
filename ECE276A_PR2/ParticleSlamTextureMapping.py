import numpy as np
from  auxiliary import *
import load_data as ld
import p2_utils as util
import matplotlib.pyplot as plt
import cv2
class ParticleFilterSlam:

    def __init__(self,numParticle, N_threshold, resolution, xmin, ymin, xmax, ymax, xRange, yRange):
        self.num_particle=numParticle
        self.N_threshold=N_threshold
        self.resolution=resolution
        self.xmin=xmin
        self.ymin=ymin
        self.xmax=xmax
        self.ymax=ymax
        self.xRange=xRange
        self.yRange=yRange

    def getWorldFrameLaserMap(self,laserScan,robotPose, headAngles):
        lidarScanAngles = np.arange(-135, 135.25, 0.25) * np.pi / 180.0
        lidarScanAngles = np.reshape(lidarScanAngles, (1, 1081))

        laserScan, lidarScanAngles = filter_laser_scan(laserScan, lidarScanAngles)
        xlf = np.multiply(laserScan, np.cos(lidarScanAngles))
        xlf = xlf.reshape((xlf.shape[0], 1))

        ylf = np.multiply(laserScan, np.sin(lidarScanAngles))
        ylf = ylf.reshape((ylf.shape[0], 1))

        zlf = np.zeros((xlf.shape[0], 1))  # Set the z value to zero
        ones = np.ones((xlf.shape[0], 1))


        lidarData = np.concatenate((xlf, ylf, zlf, ones), axis=1)

        #Convert to world frame
        lidarDataWorldFrame = lidar2world(robotPose, headAngles, lidarData)
        lidarDataWorldFrame=lidarDataWorldFrame.transpose()

        # Filter the laser that hits the ground
        indexGround = np.logical_and((lidarDataWorldFrame[2, :] > 0.1),(np.absolute(lidarDataWorldFrame[0, :]) <self.xmax),(np.absolute(lidarDataWorldFrame[1, :]) <self.ymax))
        lidarDataXY = lidarDataWorldFrame[0: 2, indexGround]
        return lidarDataXY

    def mapping(self, laser_scan, robotPose, headAngles, MAP):
        #Specify the laser scan
        lidarScanAngles = np.arange(-135, 135.25, 0.25) * np.pi / 180.0
        lidarScanAngles=np.reshape(lidarScanAngles,(1,1081))

        #Get rid of the lidar data that are too far or too close
        laser_scan, lidarScanAngles =filter_laser_scan(laser_scan,lidarScanAngles)

        #Get the x,y,z in lidar frame
        xlf=np.multiply(laser_scan,np.cos(lidarScanAngles))
        xlf=xlf.reshape((xlf.shape[0],1))
        ylf=np.multiply(laser_scan,np.sin(lidarScanAngles))
        ylf = ylf.reshape((ylf.shape[0], 1))
        zlf=np.zeros((xlf.shape[0],1))
        ones=np.ones((xlf.shape[0],1))
        lidarData=np.concatenate((xlf,ylf,zlf,ones),axis=1)

        #Convert the lidar frame to world frame
        lidarDataWorldFrame=lidar2world(robotPose,headAngles,lidarData)
        lidarDataWorldFrame=lidarDataWorldFrame.transpose()
        #Filter the laser that hits the ground
        indexGround = np.logical_and((lidarDataWorldFrame[2, :] > 0.1),(np.absolute(lidarDataWorldFrame[0, :]) <self.xmax),(np.absolute(lidarDataWorldFrame[1, :]) <self.ymax))

        #Convert positions to map pixels
        posX_map = (np.ceil((lidarDataWorldFrame[0, indexGround] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
        posY_map = (np.ceil((lidarDataWorldFrame[1, indexGround] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1)

        x_sensor = (np.ceil((robotPose[0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
        y_sensor = (np.ceil((robotPose[1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1)

        x_occupied = np.concatenate([posX_map, [x_sensor]])
        y_occupied = np.concatenate([posY_map, [y_sensor]])

        #Update the map
        MAP['log_map'][posX_map, posY_map] += 2 * np.log(9)
        polygon = np.zeros((MAP['sizey'], MAP['sizex']))
        occupied_ind = np.vstack((y_occupied, x_occupied)).T
        cv2.drawContours(image=polygon, contours=[occupied_ind], contourIdx=0, color=np.log(1.0 / 9), thickness=-1)
        MAP['log_map'] += polygon
        occupied = MAP['log_map'] > 0
        empty = MAP['log_map'] < 0
        route = (MAP['show_map'][:, :, 0] == 255)
        MAP['map'][occupied] = 1
        MAP['show_map'][occupied, :] = 0
        MAP['show_map'][np.logical_and(empty, ~route), :] = 1
        return MAP

    def textureMapping(self, rbgImage, depthImage, particleUpdated, head_angles, MAP):

        depthImageP=depthImage2depthFrame(depthImage)
        pixelsImage=depthFrame2imageFrame(depthImageP,rbgImage)
        imageWorld=pixels2world(particleUpdated,head_angles,depthImageP)

        indexGround = np.logical_and((imageWorld[:, 2] > 0.1))

        #Convert positions to map pixels
        posX_map = (np.ceil((imageWorld[0, indexGround] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
        posY_map = (np.ceil((imageWorld[1, indexGround] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1)

        MAP['show_map'][posX_map, posY_map, pixelsImage[indexGround,:]]
        return MAP
    def prediction(self,particles, deltaPose):
        #Add the noise
        particles = particles + deltaPose+np.array([1,1,10])*np.random.normal(0, 1e-3,)
        particles=particles.reshape((3,))
        return particles

    def update(self,weight,MAP,laser,robotPose,headAngles):
        # Single Particle Update
        laserScan = laser['scan']
        #particle=np.array([0,0])
        y=self.getWorldFrameLaserMap(laserScan, robotPose, headAngles)
        y = np.concatenate([y, np.zeros((1,y.shape[1]))], axis=0)
        x_im = np.arange(self.xmin, self.xmax + self.resolution, self.resolution)  # x-positions of each pixel of the map
        y_im = np.arange(self.ymin, self.ymax + self.resolution, self.resolution)  # y-positions of each pixel of the map
        correlation = util.mapCorrelation(MAP['map'], x_im, y_im, y[0: 3, :], self.xRange, self.yRange)

        ind = np.argmax(correlation)
        indx=ind/3
        indy=ind%3
        corr = correlation[indx.astype(int), indy.astype(int)]
        robotPose[0]+= self.xRange[indx.astype(int)]
        robotPose[1]+= self.yRange[indy.astype(int)]
        return robotPose, corr

    def reSample(self,particles,weights):
        #Stratified and Systematic Resampling
            particleResample = np.zeros((self.num_particle, 3))
            c, j = weights[0], 0
            for k in range(self.num_particle):
                u = np.random.uniform(0, 1.0 / self.num_particle)
                beta = u + k * (1.0 / self.num_particle)
                while beta > c:
                    j = j + 1
                    c = c + weights[j]
                particleResample[k, :] = particles[j, :]
            weights = np.einsum('..., ...', 1.0 / self.num_particle, np.ones((self.num_particle, 1)))
            return particleResample, weights

    def slam(self,lidar,joint,rgbImagesList, depthImagesList):
        # Initialize particles and weights
        N=self.num_particle
        particles = np.zeros((N, 3))
        weights = (1.0 / N) * np.ones((N, 1))

        # Generate map
        MAP = generateMap(self.resolution, self.xmin, self.ymin, self.xmax, self.ymax)


        head_angles = joint['head_angles']
        #deltaInitialPose = getPose(lidar)

        #Get the time stamps of the head angles
        timeStamps = joint['ts']

        laser0 = lidar[0]
        laser_scan = laser0['scan']
        indexInitialAngle = np.argmin(np.absolute(timeStamps - laser0['t'][0][0]))

        MAP = self.mapping(laser_scan, particles[0,:], head_angles[:,indexInitialAngle],MAP)
        timeLength = len(lidar)
        bestParticles=[]
        lidarTime=[]
        anglesParticles=[]
        for i in range(1,100):
            #deltaPose= deltaInitialPose[:,i]
            laseri=lidar[i]
            indexAngle = np.argmin(np.absolute(timeStamps - laseri['t']))
            lidarTime.append(laseri['t'])
            anglesParticles.append(head_angles[:,indexAngle])
            particleUpdated=[]
            weightsUpdated=[]
            for k in range(N):
                particlePredicted=self.prediction(particles[k,:],laseri['delta_pose'])
                particlePredicted, corr =self.update(weights[k],MAP,laseri, particlePredicted,head_angles[:,indexAngle])
                wtmp=np.log(weights[k])+corr
                particleUpdated.append(particlePredicted)
                weightsUpdated.append(wtmp)
            particleUpdated=np.asarray(particleUpdated)
            weightsUpdated=np.asarray(weightsUpdated)

            wtmp_max=weightsUpdated[np.argmax(weightsUpdated)]
            lse=np.log(np.sum(np.exp(weightsUpdated - wtmp_max)))
            weightsUpdated = weightsUpdated - wtmp_max - lse

            weights = np.exp(weightsUpdated)

            #Using the largest weight for mapping
            indexLargest = weights.argmax()
            MAP = self.mapping(laseri['scan'], particleUpdated[indexLargest, :], head_angles[:, indexAngle], MAP)

            bestParticles.append(particleUpdated[indexLargest,:])

            #Resample the particles
            N_eff = 1 / np.sum(np.square(weights))
            if N_eff < self.N_threshold:
                particles, weights= self.reSample(particleUpdated,weights)

            if(i%1000==0):
                plt.figure()
                plt.imshow(MAP['show_map'], cmap="hot")
                plt.pause(0.05)
                plt.show()
        bestParticles=np.asarray(bestParticles)
        x_r = (np.ceil((bestParticles[:, 0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
        y_r = (np.ceil((bestParticles[:, 1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1)
        plt.plot(y_r,x_r)

        lidarTime=np.asarray(lidarTime)
        anglesParticles=np.asarray(anglesParticles)
        # Texture Mapping
        for i in range(len(depthImagesList)):
            indexAngleTeMa= np.argmin(np.absolute(lidarTime -depthImagesList[i]['t']))
            MAP = self.textureMapping(rgbImagesList[i]['image'],depthImagesList[i]['depth'],bestParticles[indexAngleTeMa,:],anglesParticles[indexAngleTeMa,:],MAP)
            if (i % 1000 == 0):
                plt.figure()
                plt.imshow(MAP['show_map'], cmap="hot")
                plt.pause(0.05)
                plt.show()
        return MAP, bestParticles


if __name__ == "__main__":
    #Specify map properties
    resolution = .05
    xmin = -40
    ymin = -40
    xmax = 40
    ymax = 40
    numParticle=10
    N_threshold = 35
    yRange = np.arange(-resolution,resolution+0.01, resolution)
    xRange = np.arange(-resolution,resolution+0.01, resolution)
    depthFolder='C:/Users/Baoqian Wang/Google Drive/cam/DEPTH_0'
    rgbFolder='C:/Users/Baoqian Wang/Google Drive/cam/RGB_0'
    depthImageList=ld.get_depth(depthFolder)
    rgbImageList=ld.get_rgb(rgbFolder)
    #Initialize ParticleSlam object
    particleSlam=ParticleFilterSlam(numParticle, N_threshold, resolution, xmin, ymin, xmax, ymax, xRange, yRange)

    # Get lidar data
    lidar=ld.get_lidar("./lidar/train_lidar0")

    # Get the head angles data
    joint = ld.get_joint("./joint/train_joint0")
    IRcalibration=ld.getIRCalib()
    # Run the SLAM
    MAP, bestParticles=particleSlam.slam(lidar, joint,rgbImageList,depthImageList)

    #fig1 = plt.figure(2)
    #plt.imshow(MAP['show_map'], cmap = "hot")
    #plt.show()