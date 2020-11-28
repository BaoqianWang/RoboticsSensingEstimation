import numpy as np
from utils import *
from scipy.linalg import expm
from scipy.linalg import inv
from matplotlib import pyplot

def predictionEKF(currentPose, linearV,rotationalV,tau):
	# Define the noise
	noise={}
	noise['mean']=0*np.identity(4)
	noise['var']=5*np.identity(6)

	uHatT=np.vstack((np.hstack((hatMap(rotationalV), linearV.reshape(3,1))),np.array([0,0,0,0])))
	uCurveHatT=np.vstack((np.hstack((hatMap(rotationalV), hatMap(linearV))),np.hstack((np.zeros((3,3)),hatMap(rotationalV)))))


	#Predict the next pose
	currentPose['mean']=np.dot(expm(-float(tau)*uHatT),currentPose['mean'])
	currentPose['var']=expm(-float(tau)*uCurveHatT)*currentPose['var']*expm(-float(tau)*uCurveHatT).T + noise['var']


def updateEKF(currentFeatureWorld, currentPose, features, K, b, cam_T_imu):

	# Intrinsic parameter matrix
	M = np.hstack((np.vstack((K[0:2, 0:3], K[0:2, 0:3])), np.array([0, 0, -K[0, 0] * b, 0]).reshape(4, -1)))

	# Noise
	V=5
	numFeatures = features.shape[1]
	H=[]
	Z=[]
	Zhat=[]

	for i in range(numFeatures):
		# The feature is not valid
		if (features[0, i] == -1):
			continue

		q = np.matmul(cam_T_imu, np.matmul(currentPose['mean'], currentFeatureWorld['mean'][i, :]))
		Hi = np.matmul(M, np.matmul(piDerivative(q), np.matmul(cam_T_imu, circleHat(np.matmul(currentPose['mean'],currentFeatureWorld['mean'][i, :])))))
		zhat = M.dot(piFunction(q))
		H.append(Hi)
		Z.append(features[:,i])
		Zhat.append(zhat)
	numValidFeatures=len(H)
	H=np.asarray(H)
	Z=np.asarray(Z)
	Zhat=np.asarray(Zhat)
	H=np.reshape(H,(H.shape[0]*H.shape[1],H.shape[2]))
	Z = np.reshape(Z, (Z.shape[0] * Z.shape[1], 1))
	Zhat = np.reshape(Zhat, (Zhat.shape[0] * Zhat.shape[1], 1))

	invPart=np.linalg.pinv(H.dot(currentPose['var']).dot(H.T) + V * np.identity(numValidFeatures * 4))
	# Calculate the Kalman Gain
	Kgain = currentPose['var'].dot(H.T).dot(invPart)
	currentPose['mean'] = expm(seHatMap(Kgain.dot(Z-Zhat))).dot(currentPose['mean'])
	currentPose['var'] = np.dot((np.identity(6) - np.dot(Kgain, H)),currentPose['var'])


def mappingEKF(currentFeatureWorld, currentPose, previousFeatures, features, K, b, cam_T_imu):
	# Projection matrix
	P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
	# Intrinsic parameter matrix
	M = np.hstack((np.vstack((K[0:2, 0:3], K[0:2, 0:3])), np.array([0, 0, -K[0, 0] * b, 0]).reshape(4, -1)))

	# Noise variance
	V=2

	numFeatures=features.shape[1]
	for i in range(numFeatures):
		# The feature is not valid
		if (features[0,i]==-1):
			continue

		if previousFeatures[0, i] == -1 and features[0, i] != -1 and currentFeatureWorld['mean'][i, 0] == 0.0001 and \
				currentFeatureWorld['mean'][i, 1] == 0.0001 and currentFeatureWorld['mean'][i, 2] == 0.0001:
			currentFeatureWorld['mean'][i, :] = np.dot(worldTimu(currentPose['mean']), np.dot(np.linalg.inv(cam_T_imu), np.hstack((K[0, 0] * b * np.dot(
				np.linalg.inv(K), np.hstack((features[:, i][0:2], 1))) / (features[:, i][0] - features[:, i][2]), 1))))
			continue

		q=np.matmul(cam_T_imu,np.matmul(currentPose['mean'],currentFeatureWorld['mean'][i,:]))
		H = np.matmul(M, np.matmul(piDerivative(q), np.matmul(cam_T_imu, np.matmul(currentPose['mean'], P))))
		zHat=M.dot(piFunction(q))

		# Calculate the Kalman Gain
		Kgain=currentFeatureWorld['var'][i,:,:].dot(H.T).dot(np.linalg.pinv(H.dot(currentFeatureWorld['var'][i,:,:]).dot(H.T)+V*np.identity(4)))
		currentFeatureWorld['mean'][i,:]=currentFeatureWorld['mean'][i,:] + np.dot(P,np.dot(Kgain,(features[:,i] - zHat)))
		currentFeatureWorld['var'][i,:,:]=np.dot((np.identity(3) - np.dot(Kgain,H)),currentFeatureWorld['var'][i,:,:])

if __name__ == '__main__':

	# Set the initial pose
	currentPose={}
	currentPose['mean']=np.identity(4)
	currentPose['var']=0.005*np.identity(6)



	filename = "./data/0034.npz"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

	# Trajectory
	poseTrajectory=[]

	# Set the initial features in world frame
	numFeatures=features.shape[1]
	currentFeatureWorld={}
	currentFeatureWorld['mean']=0.0001*np.ones((numFeatures,4))
	currentFeatureWorld['var']=np.zeros((numFeatures,3,3))

	for k in range(numFeatures):
		currentFeatureWorld['var'][k,:,:]=0.1*np.identity(3)


	timeLength=t.shape[1]
	for i in range(1,timeLength-1):
		tau=np.abs(t[0,i+1]-t[0,i])
	# (a) IMU Localization via EKF Prediction
		#predictionEKF(currentPose,linear_velocity[:,i],rotational_velocity[:,i],tau)

	# (b) Landmark Mapping via EKF Update
		#mappingEKF(currentFeatureWorld, currentPose, features[:,:,i-1],features[:,:,i],K, b, cam_T_imu)

	# (c) Visual-Inertial SLAM
		#Uncomment the following three lines  to perform  Visual-Inertial SLAM

		predictionEKF(currentPose, linear_velocity[:, i], rotational_velocity[:, i], tau)
		updateEKF(currentFeatureWorld, currentPose, features[:, :, i], K, b, cam_T_imu)
		mappingEKF(currentFeatureWorld, currentPose, features[:, :, i - 1], features[:, :, i], K, b, cam_T_imu)

		poseTrajectory.append(worldTimu(currentPose['mean']).T)
	poseTrajectory=np.asarray(poseTrajectory)
	#poseTrajectory
	test=poseTrajectory.T

	# You can use the function below to visualize the robot pose over time
	# Visualize the trajectory and landmark features
	visualize_trajectory_environment_2d(test,currentFeatureWorld['mean'], filename,show_ori=True)
	# Visualize the trajectory without landmark features
	visualize_trajectory_2d(test, filename, show_ori=True)