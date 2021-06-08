from controller import Supervisor
import os
import numpy as np



os.environ['WEBOTS_ROBOT_NAME'] = 'Robot0'
robot = Supervisor()
name = robot.getName()
timestep = int(robot.getBasicTimeStep())  # ms, default = 32


leftMotor = robot.getMotor('left wheel')
rightMotor = robot.getMotor('right wheel')
# leftMotor1 = robot.getMotor('front left wheel')
# rightMotor1 = robot.getMotor('front right wheel')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
# leftMotor1.setPosition(float('inf'))
# rightMotor1.setPosition(float('inf'))
leftMotor.setVelocity(1/0.0975) #1/0.0975
rightMotor.setVelocity(1/0.0975)
# leftMotor1.setVelocity(1) #1/0.0975
# rightMotor1.setVelocity(1)



if name=="Robot0":
    lidar = robot.getLidar('Lidar0')
    print('1111111111111111')
else:
    lidar = robot.getLidar('Lidar2')
    print("2222222222222222222222")
lidar.enable(timestep)
lidar.enablePointCloud()
lidar.setFrequency(4)
robot.getFromDef("Robot0").getField("rotation").setSFRotation([0,1,0,2])


# robot.getFromDef("Robot0").getField("translation").setSFVec3f([2, 2, 2])
# print('sss',lidar.getRangeImage())
print(robot.step(100))
while robot.step(100) != -1:
    info = lidar.getRangeImage()
    info = np.array(info)
    rotation = robot.getFromDef("Robot0").getField("rotation").getSFRotation()
    print(rotation)

#     if name=="Robot0":
#         # print('ssss')
#         position0 = robot.getFromDef("Robot0").getPosition()
#     rotation = robot.getFromDef("Robot0").getField("rotation").getSFRotation()
#     print(rotation)
#         print('position0', position0)
#         velocity = leftMotor.getVelocity()
#         # position1 = robot.getFromDef("Robot1").getPosition()
#         # print('position1:', position1)
#         #
#         # position_Pedestrian0 = robot.getFromDef("Pedestrian0").getPosition()
#         # print('Pedestrian0:', position_Pedestrian0)
#
#         # print(velocity)
#         # robot.getFromDef("pioneer1").getField("translation").setSFVec3f([0, 0, 0])
#     else:
#         position = robot.getFromDef("pioneer2").getPosition()
#         velocity = leftMotor.getVelocity()
#         robot.getFromDef("Robot0").getField("rotation").setSFRotation([0, 1, 0, 0])
#         robot.step(1)
#         # robot.getFromDef("robot-1").getField("translation").setSFVec3f([-0.2, 0, 0])
#         # print(velocity)
#     # print(position)
#     pass

