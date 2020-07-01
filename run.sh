debug_arg=""
if [ $# -gt 1 ]; then
          if [ "$1" == "--debug" ]; then
                            debug_arg="-d $2"
          fi
fi
CUDA_VISIBLE_DEVICES=0,1 bert -c data/corpus.small -v data/vocab.small -o bert.model $debug_arg
