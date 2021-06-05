import os

while(True):
    os.system("kill -9 $(ps -ef|grep ppo_webots1|gawk '$0 !~/grep/ {print $2}' |tr -s '\n' ' ')")
    os.system("mpiexec -np 1 timeout 20m python ppo_webots1.py  -nr 1 -np 0")
