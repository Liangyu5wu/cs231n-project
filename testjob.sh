#!/bin/bash

source /sdf/data/atlas/u/liangyu/dSiPM/DREAMSim/setup.sh

SRC_DIR="/sdf/data/atlas/u/liangyu/dSiPM/DREAMSim/sim/src"
OUTER_DIR="/sdf/data/atlas/u/liangyu/dSiPM/DREAMSim/test"
BUILD_DIR="/sdf/data/atlas/u/liangyu/dSiPM/DREAMSim/sim"
STEPPING_FILE="B4bSteppingAction.cc"
LUSTRE_DIR="/fs/ddn/sdf/group/atlas/d/liangyu/dSiPM/cs231n"

echo "Starting scanning..."

TOTAL_EVENTS=2000
EVENTS_PER_JOB=10
NUM_JOBS=200
GUN_ENERGY_MIN=30
GUN_ENERGY_MAX=30
PARTICLE_NAME="e-"


    
cd $SRC_DIR
# sed -i "822s/.*/  if (!(isCoreS || isCoreC || isCladS || isCladC) || rodNumber < 35 || rodNumber > 55 || layerNumber < 32 || layerNumber > 50)/" $STEPPING_FILE
sed -i "822s/.*/  if (!(isCoreS || isCoreC || isCladS || isCladC) || rodNumber < 20 || rodNumber > 60 || layerNumber < 15 || layerNumber > 65)/" $STEPPING_FILE
echo "$STEPPING_FILE 822 line gets modified!"

TEMP_SCRIPT=$(mktemp)
cat > $TEMP_SCRIPT << EOF
#!/bin/bash
rm -rf /sdf/data/atlas/u/liangyu/dSiPM/DREAMSim/sim/build
source /sdf/data/atlas/u/liangyu/dSiPM/DREAMSim/setup9.sh
cd $BUILD_DIR
mkdir build
cd build
cmake ..
make -j 4
EOF
chmod +x $TEMP_SCRIPT
singularity exec --bind=/cvmfs,/sdf,/fs,/lscratch /cvmfs/atlas.cern.ch/repo/containers/fs/singularity/x86_64-almalinux9 $TEMP_SCRIPT
rm $TEMP_SCRIPT
    
echo "========================================="

cd $OUTER_DIR
WORK_DIR="${PARTICLE_NAME}_E${GUN_ENERGY_MIN}GeV_${TOTAL_EVENTS}jobs"
mkdir -p $WORK_DIR
cd $WORK_DIR


for ((job_id=1; job_id<=NUM_JOBS; job_id++)); do
  JOB_SCRIPT="job_E${GUN_ENERGY_MIN}_${job_id}.sh"
  PAYLOAD_SCRIPT="myJobPayload_E${GUN_ENERGY_MIN}_job${job_id}.sh"

  cat > $PAYLOAD_SCRIPT << EOF
#!/bin/bash
source /sdf/data/atlas/u/liangyu/dSiPM/DREAMSim/setup9.sh
cd /sdf/data/atlas/u/liangyu/dSiPM/DREAMSim/sim/build
./exampleB4b -b paramBatch03_single.mac -jobName ${PARTICLE_NAME}_job -runNumber 1 -runSeq ${job_id} -numberOfEvents ${EVENTS_PER_JOB} -eventsInNtupe 100 -gun_particle ${PARTICLE_NAME} -gun_energy_min ${GUN_ENERGY_MIN} -gun_energy_max ${GUN_ENERGY_MAX} -sipmType 1
echo "Job ${job_id} for energy=${GUN_ENERGY_MIN}GeV completed!"
EOF
  
      
  cat > $JOB_SCRIPT << EOF
#!/bin/bash
#
#SBATCH --account=atlas:default
#SBATCH --partition=roma
#SBATCH --job-name=RE${GUN_ENERGY_MIN}_${job_id}
#SBATCH --output=output_E${GUN_ENERGY_MIN}_job${job_id}-%j.txt
#SBATCH --error=error_E${GUN_ENERGY_MIN}_job${job_id}-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4g
#SBATCH --time=4:00:00

unset KRB5CCNAME
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
export ALRB_CONT_CMDOPTS="-B /sdf,/fs,/lscratch"
export ALRB_CONT_RUNPAYLOAD="source /sdf/data/atlas/u/liangyu/dSiPM/DREAMSim/test/${WORK_DIR}/${PAYLOAD_SCRIPT}"

source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh -c el9 –pwd $PWD
EOF

  chmod +x $JOB_SCRIPT
  chmod +x $PAYLOAD_SCRIPT
  sbatch $JOB_SCRIPT
      
  echo "Submitted job ${job_id}"
      
  if (( job_id % 50 == 0 )); then
    echo "Pausing for 3 seconds to prevent overwhelming the scheduler..."
    sleep 3
  fi
done
    
echo "All ${NUM_JOBS} jobs submitted"
echo "========================================="

echo "Waiting for all jobs to complete..."
    
while true; do
  JOB_PREFIX="RE${GUN_ENERGY_MIN}"
  RUNNING_JOBS=$(squeue -u $USER -h -o "%.15j" | grep "${JOB_PREFIX}_" | wc -l)
      
  if [ $RUNNING_JOBS -eq 0 ]; then
    echo "All jobs for $JOB_PREFIX have completed. Continuing with the next parameter set."
    break
  else
    echo "$(date): Still waiting for $RUNNING_JOBS jobs with prefix $JOB_PREFIX to complete..."
    sleep 30
  fi
done


LUSTRE_SUBDIR="${LUSTRE_DIR}/${PARTICLE_NAME}_E${GUN_ENERGY_MIN}-${GUN_ENERGY_MAX}_${TOTAL_EVENTS}"
mkdir -p ${LUSTRE_SUBDIR}
echo "Moving ROOT files to Lustre filesystem at ${LUSTRE_SUBDIR}..."

ROOT_FILES_PATH="/sdf/data/atlas/u/liangyu/dSiPM/DREAMSim/sim/build"
ROOT_FILE_PATTERN="mc_${PARTICLE_NAME}_job_run1_*_Test_${EVENTS_PER_JOB}evt_${PARTICLE_NAME}_${GUN_ENERGY_MIN}_${GUN_ENERGY_MAX}.root"
    
find ${ROOT_FILES_PATH} -name "${ROOT_FILE_PATTERN}" -exec mv {} ${LUSTRE_SUBDIR}/ \;
MOVED_FILES_COUNT=$(ls -1 ${LUSTRE_SUBDIR}/${ROOT_FILE_PATTERN} 2>/dev/null | wc -l)
echo "Moved ${MOVED_FILES_COUNT} ROOT files to Lustre filesystem"


rm -rf $OUTER_DIR/$WORK_DIR

echo "Scanning and job submission completed!"
