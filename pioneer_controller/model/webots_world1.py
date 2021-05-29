import time
import copy
import tf
import numpy as np




class WebotsWorld():
    def __init__(self, beam_num, index, robot, num_robot, num_pedestrian):
        self.index = index
        self.num_robot = num_robot
        self.num_pedestrian = num_pedestrian
        self.robot_name = 'Robot' + str(index)
        self.lidar_name = 'Lidar' + str(index)

        self.robot_radius = 0.26
        self.pedestrian_radius = 0.3

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

        # # -----------Publisher and Subscriber-------------
        # cmd_vel_topic = 'robot_' + str(index) + '/cmd_vel'
        # self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
        #
        # cmd_pose_topic = 'robot_' + str(index) + '/cmd_pose'
        # self.cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=2)
        #
        # object_state_topic = 'robot_' + str(index) + '/base_pose_ground_truth'
        # self.object_state_sub = rospy.Subscriber(object_state_topic, Odometry, self.ground_truth_callback)
        #
        # laser_topic = 'robot_' + str(index) + '/base_scan'
        #
        # self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback)
        #
        # odom_topic = 'robot_' + str(index) + '/odom'
        # self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)
        #
        # crash_topic = 'robot_' + str(index) + '/is_crashed'
        # self.check_crash = rospy.Subscriber(crash_topic, Int8, self.crash_callback)
        #
        #
        # self.sim_clock = rospy.Subscriber('clock', Clock, self.sim_clock_callback)
        #
        # # -----------Service-------------------
        # self.reset_stage = rospy.ServiceProxy('reset_positions', Empty)


        # # Wait until the first callback
        # self.speed = None
        # self.state = None
        # self.speed_GT = None
        # self.state_GT = None
        # while self.scan is None or self.speed is None or self.state is None\
        #         or self.speed_GT is None or self.state_GT is None:
        #     pass

        # rospy.sleep(1.)
        # # What function to call when you ctrl + c
        # rospy.on_shutdown(self.shutdown)


    # def ground_truth_callback(self, GT_odometry):
    #     Quaternious = GT_odometry.pose.pose.orientation
    #     Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
    #     self.state_GT = [GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y, Euler[2]]
    #     v_x = GT_odometry.twist.twist.linear.x
    #     v_y = GT_odometry.twist.twist.linear.y
    #     v = np.sqrt(v_x**2 + v_y**2)
    #     self.speed_GT = [v, GT_odometry.twist.twist.angular.z]
    #
    # def laser_scan_callback(self, scan):
    #     self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
    #                        scan.scan_time, scan.range_min, scan.range_max]
    #     self.scan = np.array(scan.ranges)
    #     self.laser_cb_num += 1
    #
    #
    # def odometry_callback(self, odometry):
    #     Quaternions = odometry.pose.pose.orientation
    #     Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
    #     self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
    #     self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]
    #
    # def sim_clock_callback(self, clock):
    #     self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.
    #
    # def crash_callback(self, flag):
    #     self.is_crashed = flag.data

    def get_self_stateGT(self):
        Quaternious = self.robot.getFromDef(self.robot_name).getField("rotation").getSFRotation()
        Euler = tf.transformations.euler_from_quaternion([Quaternious[1], Quaternious[2], Quaternious[3], Quaternious[0]])
        # self.state_GT = [self.robot.getFromDef(self.robot_name).getPosition()[0],self.robot.getFromDef("Robot0").getPosition()[2], Euler[2]]
        # return self.state_GT
        return [self.robot.getFromDef(self.robot_name).getPosition()[0],self.robot.getFromDef(self.robot_name).getPosition()[2], Euler[2]]

    def get_self_speedGT(self):
        # self.speed_GT = [self.leftMotor.getVelocity(), self.rightMotor.getVelocity()]
        # return self.speed_GT
        return [self.leftMotor.getVelocity(), self.rightMotor.getVelocity()]

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
        return scan_sparse / 6.0 - 0.5


    def get_self_speed(self):
        return [self.leftMotor.getVelocity(), self.rightMotor.getVelocity()]

    # def get_self_state(self):
    #     Quaternious = self.robot.getFromDef(self.robot_name).getField("rotation").getSFRotation()
    #     Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
    #     return [self.robot.getFromDef(self.robot_name).getPosition()[0],self.robot.getFromDef("Robot0").getPosition()[2], Euler[2]]

    def get_crash_state(self, position,safe_distance=0.):
        # position = self.get_self_stateGT()
        other_robots_position = self.get_other_robots_position()
        if abs(position[0])>9.5 or abs(position[1])>9.5:
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
            dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.robot_radius - self.robot_radius
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
            robots_position.append([self.robot.getFromDef(robot).getPosition()[0],self.robot.getFromDef(robot).getPosition()[2]])
        return robots_position

    def get_pedestrians_position(self):
        pedestrians_name = []
        pedestrians_position = []
        for i in range(self.num_pedestrian):
            pedestrians_name.append('Pedestrian'+str(i))
        for pedestrian in pedestrians_name:
            pedestrians_position.append([self.robot.getFromDef(pedestrian).getPosition()[0],self.robot.getFromDef(pedestrian).getPosition()[2]])
        return pedestrians_position



    def get_local_goal(self):
        [x, y, theta] = self.get_self_stateGT()
        [goal_x, goal_y] = self.goal_point
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
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



    def generate_goal_point(self):
        [x_g, y_g] = self.generate_random_goal()
        self.goal_point = [x_g, y_g]
        [x, y] = self.get_local_goal()

        self.pre_distance = np.sqrt(x ** 2 + y ** 2)
        self.distance = copy.deepcopy(self.pre_distance)


    def get_reward_and_terminate(self, t):
        terminate = False
        laser_scan = self.get_laser_observation()
        [x, y, theta] = self.get_self_stateGT()
        [v, w] = self.get_self_speedGT()
        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)
        reward_g = (self.pre_distance - self.distance) * 2.5  # reward shaping
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
        print(reward_g, reward_c, reward_w)
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
        # move_cmd = Twist()
        # move_cmd.linear.x = action[0]
        # move_cmd.linear.y = 0.
        # move_cmd.linear.z = 0.
        # move_cmd.angular.x = 0.
        # move_cmd.angular.y = 0.
        # move_cmd.angular.z = action[1]
        # self.cmd_vel.publish(move_cmd)
        leftMotor = self.robot.getMotor('left wheel')
        rightMotor = self.robot.getMotor('right wheel')
        leftMotor.setPosition(float('inf'))
        rightMotor.setPosition(float('inf'))
        leftMotor.setVelocity(action[0])
        rightMotor.setVelocity(action[1])
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
        self.robot.getFromDef(self.robot_name).getField("translation").setSFVec3f([pose[0], 0.0949247, pose[1]])
        qtn = tf.transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')
        self.robot.getFromDef(self.robot_name).getField("rotation").setSFRotation([qtn[3], qtn[0], qtn[1], qtn[2]])
        # self.robot.step(1)

    def generate_random_pose(self):
        near_obstacle = True
        proper_origin = False
        while(near_obstacle == True and proper_origin == False):
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
            if dis_origin < 9:
                proper_origin = True
            if dis_goal < 10 and dis_goal > 8:
                proper_goal = True
            # if near_obstacle is False and proper_goal is True and proper_origin is True:
            #     break
        #print('POSITIONNNNNNNNN: ',self.init_pose,'dis_goal',dis_goal)
        return [x, y]
