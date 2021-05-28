from controller import Supervisor

robot = Supervisor()
name = robot.getName()
timestep = int(robot.getBasicTimeStep())

leftMotor = robot.getMotor('left wheel')
rightMotor = robot.getMotor('right wheel')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(1/0.0975)
rightMotor.setVelocity(1/0.0975)

if name=="Robot0":
    lidar = robot.getLidar('Lidar0')
    print('1111111111111111')
else:
    lidar = robot.getLidar('lidar2')
    print("2222222222222222222222")
lidar.enable(timestep)
lidar.enablePointCloud()
lidar.setFrequency(4)

robot.getFromDef("Robot0").getField("translation").setSFVec3f([0, 0, 5])


while robot.step(timestep) != -1:
    info = lidar.getRangeImage()
    print(info)
    if name=="Robot0":
        print('ssss')
        position = robot.getFromDef("pioneer1").getPosition()
        # robot.getFromDef("pioneer1").getField("translation").setSFVec3f([0, 0, 0])
    else:
        position = robot.getFromDef("pioneer2").getPosition()
        velocity = leftMotor.getVelocity()
        # robot.getFromDef("robot-1").getField("rotation").setSFRotation([0, 1, 0, 0])
        # robot.getFromDef("robot-1").getField("translation").setSFVec3f([-0.2, 0, 0])
        print(velocity)
    # print(position)
    pass
