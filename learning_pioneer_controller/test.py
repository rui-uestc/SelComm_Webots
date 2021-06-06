import tf
import numpy as np
# qtn = tf.transformations.quaternion_from_euler(0, 0, 1, 'rxyz')
eu = tf.transformations.euler_from_quaternion([0.787673,0.00847994,0.615995,-0.00706603])
print(eu)