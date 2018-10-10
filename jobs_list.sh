!/usr/bin/env bash
export QT_QPA_PLATFORM=offscreen
export HOME=/home/rantonello/implementer/ImplementerRNN
#export PATH=/home/gparascandolo/anaconda2/bin:$PATH
#module load cuda/8.0

echo "Launching job" $1
case "$1" in

1)  python3 $HOME/MNIST_test_variable_length.py \
    --imsize 16  \
    --epochs 100 \
    --p_dim 20 \
    --reg_lambda 0 \
    --H_lr 0.001 \
    --p_lr 0.1 \
    --lstm_size 128 \
    --meta_folder ./meta
        ;;
*) echo "Signal number $1 is not processed"
    ;;
esac