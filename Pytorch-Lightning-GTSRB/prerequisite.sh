CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$CWD")"

TRAIN_SET_PATH="http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"
if [ -f "${CWD}/GTSRB_Final_Training_Images.zip" ]
then 
    :
else
    wget ${TRAIN_SET_PATH}
fi
unzip ${CWD}/GTSRB_Final_Training_Images.zip
rm unzip ${CWD}/GTSRB_Final_Training_Images.zip

TEST_IMAGE_PATH="http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip"
if [ -f "${CWD}/GTSRB_Final_Test_Images.zip" ]
then 
    :
else
    wget ${TEST_IMAGE_PATH}
fi
unzip ${CWD}/GTSRB_Final_Test_Images.zip
rm ${CWD}/GTSRB_Final_Test_Images.zip

TEST_LABEL_PATH="http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip"
if [ -f "${CWD}/GTSRB_Final_Test_GT.zip" ]
then 
    :
else
    wget ${TEST_LABEL_PATH}
fi
unzip ${CWD}/GTSRB_Final_Test_GT.zip
rm ${CWD}/GTSRB_Final_Test_GT.zip

TRAIN_DIR="${CWD}/train_artifact"
if [ -d "${TRAIN_DIR}" ]
then 
    :
else
    mkdir -p ${TRAIN_DIR}
    mv -v ${CWD}/GTSRB/Final_Training/Images/* ${TRAIN_DIR}
fi

TEST_DIR="${CWD}/test_artifact"
if [ -d "${TEST_DIR}" ]
then 
    :
else
    mkdir -p ${TEST_DIR}
    mv -v ${CWD}/GTSRB/Final_Test/Images/* ${TEST_DIR}
    mv ${CWD}/GT-final_test.csv ${TEST_DIR}/GT-final_test.csv
fi

rm -rf ${CWD}/GTSRB


