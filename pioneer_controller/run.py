import os

while(True):
    os.system("kill -9 $(ps -ef|grep ppo_webots1|gawk '$0 !~/grep/ {print $2}' |tr -s '\n' ' ')")
    os.system("mpiexec -np 6 timeout 20m python ppo_webots1.py  -nr 6 -np 0")
