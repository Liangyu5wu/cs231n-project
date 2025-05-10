#!/bin/bash
source /sdf/data/atlas/u/liangyu/vertextiming/user.scheong.mc21_14TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.SuperNtuple.e8514_s4345_r15583.20250219_Output/setup9.sh
echo "Set up!"

# modify this line to your own working path: e.g.   cd /fs/ddn/sdf/group/supercdms/qihua/test
cd /fs/ddn/sdf/group/atlas/d/liangyu/dSiPM/cs231n

python h5builder_qh.py

echo "Job completed!"
