import time
import copy
import numpy as np

# Pioneer
WHEELS_DISTANCE = 0.33
WHEEL_RADIUS = 0.0975
ROBOT_RADIUS = 0.26

# Turtlebot3 Burge
# WHEELS_DISTANCE = 0.16
# WHEEL_RADIUS = 0.033
# ROBOT_RADIUS = 0.1

# epuck
# WHEELS_DISTANCE = 0.052
# WHEEL_RADIUS = 0.02
# ROBOT_RADIUS = 0.035


class WebotsWorld():
    def __init__(self, beam_num, index, robot, num_robot, num_pedestrian):
        self.index = index
        self.num_robot = num_robot
        self.num_pedestrian = num_pedestrian
        self.robot_name = 'Robot' + str(index)
        self.lidar_name = 'Lidar' + str(index)

        self.robot_radius = ROBOT_RADIUS
        self.pedestrian_radius = 0.15

        # os.environ['WEBOTS_ROBOT_NAME'] = self.robot_name
        self.robot = robot
        self.timestep = int(self.robot.getBasicTimeStep())

        self.leftMotor = self.robot.getMotor('left wheel')
        self.rightMotor = self.robot.getMotor('right wheel')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))

        self.beam_mum = beam_num
        self.laser_cb_num = 0
        self.scan = None

        # used in reset_world
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.

        # used in generate goal point
        self.map_size = np.array([8., 8.], dtype=np.float32)  # 20x20m
        self.goal_size = 0.5

        self.robot_value = 10.
        self.goal_value = 0.
        # self.reset_pose = None

        self.init_pose = None



        # for get reward and terminate
        self.stop_counter = 0

        self.speed = None
        self.state = None

        self.lidar = self.robot.getLidar(self.lidar_name)
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        self.lidar.setFrequency(4)


    def get_self_stateGT(self):
        angle = self.robot.getFromDef(self.robot_name).getField("rotation").getSFRotation()
        # Euler = tf.transformations.euler_from_quaternion([Quaternious[1], Quaternious[2], Quaternious[3], Quaternious[0]])
        # self.state_GT = [self.robot.getFromDef(self.robot_name).getPosition()[0],self.robot.getFromDef("Robot0").getPosition()[2], Euler[2]]
        # return self.state_GT
        return [self.robot.getFromDef(self.robot_name).getPosition()[2],self.robot.getFromDef(self.robot_name).getPosition()[0], angle[3]]

    def get_self_speedGT(self):
        # self.speed_GT = [self.leftMotor.getVelocity(), self.rightMotor.getVelocity()]
        # return self.speed_GT
        v_left = self.leftMotor.getVelocity() * WHEEL_RADIUS
        v_right = self.rightMotor.getVelocity() * WHEEL_RADIUS
        return [(v_left + v_right)/2, (v_right - v_left)/WHEELS_DISTANCE]

    def get_laser_observation(self):
        scan = self.lidar.getRangeImage()
        scan = np.array(scan)
        scan[np.isnan(scan)] = 6.0
        scan[np.isinf(scan)] = 6.0
        raw_beam_num = len(scan)
        sparse_beam_num = self.beam_mum
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []
        index = 0.
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan[int(index)])
            index += step
        sparse_scan_right = []
        index = raw_beam_num - 1.
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step
        scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)
        # print(scan_sparse / 6.0 - 0.5)
        return scan_sparse / 6.0 - 0.5


    def get_self_speed(self):
        v_left = self.leftMotor.getVelocity() * WHEEL_RADIUS
        v_right = self.rightMotor.getVelocity() * WHEEL_RADIUS
        return [(v_left + v_right) / 2, (v_right - v_left) / WHEELS_DISTANCE]


    def get_crash_state(self, position,safe_distance=0.):
        # position = self.get_self_stateGT()
        other_robots_position = self.get_other_robots_position()
        if np.sqrt(position[0] ** 2 + position[1] ** 2) > 9.5:
            is_crashed = True
            return is_crashed
        for other in other_robots_position:
            dx = position[0] - other[0]
            dy = position[1] - other[1]
            dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.robot_radius - self.robot_radius - safe_distance
            # print(position, other,dist)
            if dist < 0:
                is_crashed = True
                return is_crashed

        pedestrian_position = self.get_pedestrians_position()
        for other in pedestrian_position:
            dx = position[0] - other[0]
            dy = position[1] - other[1]
            dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.pedestrian_radius - self.robot_radius
            if dist < 0:
                is_crashed = True
                return is_crashed
        is_crashed = False
        return is_crashed


    def get_other_robots_position(self):
        robots_name = []
        robots_position = []
        for i in range(self.num_robot):
            if i != self.index:
                robots_name.append('Robot'+str(i))
        for robot in robots_name:
            robots_position.append([self.robot.getFromDef(robot).getPosition()[2],self.robot.getFromDef(robot).getPosition()[0]])
        return robots_position

    def get_pedestrians_position(self):
        pedestrians_name = []
        pedestrians_position = []
        for i in range(self.num_pedestrian):
            pedestrians_name.append('Pedestrian'+str(i))
        for pedestrian in pedestrians_name:
            pedestrians_position.append([self.robot.getFromDef(pedestrian).getPosition()[2],self.robot.getFromDef(pedestrian).getPosition()[0]])
        return pedestrians_position



    def get_local_goal(self):
        [x, y, theta] = self.get_self_stateGT()
        [goal_x, goal_y] = self.goal_point
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        # print('[x, y, theta]', [x, y, theta])
        # print('[goal_x, goal_y]',[goal_x, goal_y])
        # print('[local_x, local_y]',[local_x, local_y])
        return [local_x, local_y]

    def get_position(self):
        [x, y, theta] = self.get_self_stateGT()
        return [x, y, theta]

    def reset_world(self):
        # self.reset_stage()
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.
        self.start_time = time.time()

    def draw_rects(self, coords):
        for i, goal in enumerate(coords):
            exisiting_goal = self.robot.getFromDef("GOAL_{}".format(self.index))
            if not exisiting_goal is None:
                exisiting_goal.remove()
            # print(goal)
            z, x = goal   # all x and y is bule and red,except this function
            goal_string = """
                    DEF GOAL_%d Shape{
                        appearance Appearance{
                            material Material {
                                diffuseColor 0 1 0
                                emissiveColor 0 1 0
                            }
                        }
                        geometry DEF GOAL_FACE_SET IndexedFaceSet {
                            coord Coordinate {
                                point [ %f 0.05 %f, %f 0.05 %f, %f 0.05 %f, %f 0.05 %f ]
                            }
                            coordIndex [ 3 2 1 0 -1]
                        }
                        isPickable FALSE
                    }
                    """ % (self.index, x - 0.05, z - 0.05, x + 0.05, z - 0.05, x + 0.05, z + 0.05, x - 0.05, z + 0.05)
            self.robot.getRoot().getField("children").importMFNodeFromString(-1, goal_string)
            # self.robot.step(1)

    def generate_goal_point(self):
        [x_g, y_g] = self.generate_random_goal()
        self.goal_point = [x_g, y_g]
        [x, y] = self.get_local_goal()

        self.pre_distance = np.sqrt(x ** 2 + y ** 2)
        self.distance = copy.deepcopy(self.pre_distance)
        self.draw_rects(coords=[self.goal_point])



    def get_reward_and_terminate(self, t):
        terminate = False
        laser_scan = self.get_laser_observation()
        [x, y, theta] = self.get_self_stateGT()
        [v, w] = self.get_self_speedGT()
        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)
        reward_g = (self.pre_distance - self.distance) * 2.5 # reward shaping
        reward_c = 0
        reward_w = 0
        result = 0
        is_crash = self.get_crash_state(position=self.get_self_stateGT())

        if self.distance < self.goal_size:
            terminate = True
            reward_g = 15
            result = 'Reach Goal'

        if is_crash == 1:
            terminate = True
            reward_c = -15
            result = 'Crashed'

        if np.abs(w) >  1.05:
            reward_w = -0.1 * np.abs(w)

        if t > 170:
            terminate = True
            result = 'Time out'
        reward = reward_g + reward_c + reward_w

        return reward, terminate, result

    def reset_pose(self):
        random_pose = self.generate_random_pose()
        self.control_pose(random_pose)  #!!!!!!!!
        # [x_robot, y_robot, theta] = self.get_self_stateGT()
        # # self.robot.step(1)
        #
        #
        # # start_time = time.time()
        # while np.abs(random_pose[0] - x_robot) > 0.2 or np.abs(random_pose[1] - y_robot) > 0.2:
        #     [x_robot, y_robot, theta] = self.get_self_stateGT()
        #     self.control_pose(random_pose)
        #     print('ssssss', self.index)
        #     self.robot.step(1)
        #     print('ssssss', self.index)




    def control_vel(self, action):
        v_left = action[0] - (action[1] * WHEELS_DISTANCE) / 2.0
        v_right = action[0] + (action[1] * WHEELS_DISTANCE) / 2.0
        leftMotor = self.robot.getMotor('left wheel')
        rightMotor = self.robot.getMotor('right wheel')
        # print('wheel speed',v_left, v_right)
        leftMotor.setPosition(float('inf'))
        rightMotor.setPosition(float('inf'))
        leftMotor.setVelocity(v_left / WHEEL_RADIUS)
        rightMotor.setVelocity(v_right / WHEEL_RADIUS)
        self.robot.step(1)




    def control_pose(self, pose):
        # pose_cmd = Pose()
        assert len(pose)==3
        # pose_cmd.position.x = pose[0]
        # pose_cmd.position.y = pose[1]
        # pose_cmd.position.z = 0
        #
        # qtn = tf.transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')
        # pose_cmd.orientation.x = qtn[0]
        # pose_cmd.orientation.y = qtn[1]
        # pose_cmd.orientation.z = qtn[2]
        # pose_cmd.orientation.w = qtn[3]   # using quaternion to represent orientation
        # self.cmd_pose.publish(pose_cmd)
        self.robot.getFromDef(self.robot_name).getField("translation").setSFVec3f([pose[1], 0.095, pose[0]])
        # qtn = tf.transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')
        self.robot.getFromDef(self.robot_name).getField("rotation").setSFRotation([0, 1, 0, pose[2]])
        # self.robot.step(1)

    def generate_random_pose(self):
        near_obstacle = True
        proper_origin = False
        while(near_obstacle == True or proper_origin == False):
            near_obstacle = True
            proper_origin = False
            try_position = [np.random.uniform(-9, 9), np.random.uniform(-9, 9)]
            x = try_position[0]
            y = try_position[1]
            if not self.get_crash_state(try_position,safe_distance=0.5):
                near_obstacle = False
            dis = np.sqrt(x ** 2 + y ** 2)
            if dis<9:
                proper_origin = True
        theta = np.random.uniform(0, 2 * np.pi)
        #print(near_obstacle,'dis: ',dis)
        # print('sssssssss',[x, y, theta])
        return [x, y, theta]

    def generate_random_goal(self):
        self.init_pose = self.get_self_stateGT()
        near_obstacle = True
        proper_origin = False
        proper_goal = False
        while (not proper_origin or not proper_goal or near_obstacle):
            near_obstacle = True
            proper_origin = False
            proper_goal = False
            try_position = [np.random.uniform(-9, 9),np.random.uniform(-9, 9)]
            x = try_position[0]
            y = try_position[1]
            if not self.get_crash_state(try_position,safe_distance=0.5):
                near_obstacle = False

            dis_origin = np.sqrt(x ** 2 + y ** 2)
            dis_goal = np.sqrt((x - self.init_pose[0]) ** 2 + (y - self.init_pose[1]) ** 2)
            # print(self.init_pose,x,y)
            if dis_origin < 9:
                proper_origin = True
            if dis_goal < 6 and dis_goal > 5:
                proper_goal = True
            # if near_obstacle is False and proper_goal is True and proper_origin is True:
            #     break
        #print('POSITIONNNNNNNNN: ',self.init_pose,'dis_goal',dis_goal)
        return [x, y]
