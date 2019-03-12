export PYTHONPATH=`pwd`
MODEL=$1
python2 training_ptr_gen/decode.py $MODEL

