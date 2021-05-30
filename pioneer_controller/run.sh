kill -9 $(ps -ef|grep ppo_webots1|gawk '$0 !~/grep/ {print $2}' |tr -s '\n' ' ')
mpiexec -np 8 python ppo_webots1.py  -nr 8 -np 12
