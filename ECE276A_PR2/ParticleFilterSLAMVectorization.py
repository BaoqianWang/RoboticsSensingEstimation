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
        lidarDataWorldFrame = lidar2world(robotPose, headAngles, lidarData)
        lidarDataWorldFrame=lidarDataWorldFrame.transpose()
        indexGround = lidarDataWorldFrame[2, :] > 0.1  # Filter the laser that hits the ground
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
        indexGround = lidarDataWorldFrame[2, :] > 0.1

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

    def prediction(self,particles, deltaPose):
        #Add the noise
        particles = particles + deltaPose+np.array([1,1,10])*np.random.normal(0, 1e-2,(self.num_particle,1))

        return particles

    def update(self,weights,MAP,laser,robotPose,headAngles):
        # Single Particle Update
        laserScan = laser['scan']
        x_im = np.arange(self.xmin, self.xmax + self.resolution,
                         self.resolution)  # x-positions of each pixel of the map
        y_im = np.arange(self.ymin, self.ymax + self.resolution,
                         self.resolution)  # y-positions of each pixel of the map
        for i in range(robotPose.shape[0]):
            y=self.getWorldFrameLaserMap(laserScan, robotPose[i,:], headAngles)
            y = np.concatenate([y, np.zeros((1,y.shape[1]))], axis=0)
            correlation = util.mapCorrelation(MAP['map'], x_im, y_im, y[0: 3, :], self.xRange, self.yRange)
            ind = np.argmax(correlation)
            indx=ind/3
            indy=ind%3
            corr = correlation[indx.astype(int), indy.astype(int)]
            weights[i] = weights[i]*np.exp(corr)
        return weights

    def reSample(self,particles,weights ):
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

    def slam(self,lidar,joint):
        # Initialize particles and weights
        N=self.num_particle
        particles = np.zeros((N, 3))
        head_angles = joint['head_angles']
        deltaInitialPose = getPose(lidar)

        weights = (1.0 / N) * np.ones((N, 1))
        timeStamps = joint['ts']
        MAP = generateMap(self.resolution,self.xmin, self.ymin, self.xmax, self.ymax)
        laser0 = lidar[0]
        laser_scan = laser0['scan']
        indexInitialAngle = np.argmin(np.absolute(timeStamps - laser0['t']))

        MAP = self.mapping(laser_scan, particles[0,:], head_angles[:,indexInitialAngle],MAP)
        timeLength = len(lidar)
        bestParticles=[]
        for i in range(1,timeLength):
            deltaPose= deltaInitialPose[:,i]
            laseri=lidar[i]
            indexAngle = np.argmin(np.absolute(timeStamps - laseri['t']))

            particleUpdated=[]
            weightsUpdated=[]
            particlePredicted = self.prediction(particles, deltaPose)
            weightsUpdated = self.update(weights, MAP, laseri, particlePredicted, head_angles[:, indexAngle])
            # for k in range(N):
            #     particlePredicted=self.prediction(particles[k,:],deltaPose)
            #     weight =self.update(weights[k],MAP,laseri, particlePredicted,head_angles[:,indexAngle])
            #     particleUpdated.append(particlePredicted)
            #     weightsUpdated.append(weight)
            # particleUpdated=np.asarray(particleUpdated)
            # weightsUpdated=np.asarray(weightsUpdated)

            #Using the largest weight for mapping
            indexLargest = weightsUpdated.argmax()
            MAP = self.mapping(laseri['scan'], particleUpdated[indexLargest, :], head_angles[:, indexAngle], MAP)
            #x_r = (np.ceil((particles[indexLargest, 0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
            #y_r = (np.ceil((particles[indexLargest, 1] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1)
            #MAP['show_map'][x_r, y_r, 0] = 255

            #Resample the particles
            N_eff = 1 / np.sum(np.square(weightsUpdated))
            if N_eff < self.N_threshold:
                particles, weights= self.reSample(particleUpdated,weightsUpdated)

            if(i%100==0):
                plt.figure()
                plt.imshow(MAP['show_map'], cmap="hot")
                plt.pause(0.05)
                plt.show()
        return MAP


if __name__ == "__main__":
    #Specify map properties
    resolution = 0.05
    xmin = -40
    ymin = -40
    xmax = 40
    ymax = 40
    numParticle=80
    N_threshold = 35
    yRange = np.arange(-0.05, 0.06, 0.05)
    xRange = np.arange(-0.05, 0.06, 0.05)

    #Initialize ParticleSlam object
    particleSlam=ParticleFilterSlam(numParticle, N_threshold, resolution, xmin, ymin, xmax, ymax, xRange, yRange)

    # Get lidar data
    lidar=ld.get_lidar("./lidar/train_lidar0")

    # Get the head angles data
    joint = ld.get_joint("./joint/train_joint0")

    # Run the SLAM
    MAP=particleSlam.slam(lidar, joint)

    fig1 = plt.figure(2)
    plt.imshow(MAP['show_map'], cmap = "hot")
    plt.show()

