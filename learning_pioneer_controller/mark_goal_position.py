from controller import Supervisor
import os
import numpy as np

def draw_rects(root, coords):
    for i, goal in enumerate(coords):
        exisiting_goal = robot.getFromDef("GOAL_{}".format(i))
        if not exisiting_goal is None:
            exisiting_goal.remove()
        print(goal)
        x, z = goal
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
                """ % (i, x - 0.05, z - 0.05, x + 0.05, z - 0.05, x + 0.05, z + 0.05, x - 0.05, z + 0.05)
        robot.getRoot().getField("children").importMFNodeFromString(-1, goal_string)
        robot.step(1)



os.environ['WEBOTS_ROBOT_NAME'] = 'Robot0'

robot = Supervisor()

draw_rects(root=robot,coords=[[3,4]])

