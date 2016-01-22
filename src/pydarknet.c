
#include "pydarknet.h"
#include "network.h"
#include "parser.h"
#include "yolo.h"

typedef unsigned char uint8;

#define PYTHON_DARKNET extern DARKNET_DETECTOR_EXPORT

PYTHON_DARKNET void detect(char *config_filepath, char *weight_filepath, char **input_gpath_array,
                           int num_input, float thresh, float* results_array,
                           int verbose, int quiet)
{
    if (! quiet)
    {
        #ifdef GPU
            printf("Using GPU (CUDA)\n");
        #else
            printf("Using CPU\n");
        #endif

    }
    network net = parse_network_cfg(config_filepath, verbose);
    if(weight_filepath){
        load_weights(&net, weight_filepath);
    }

    int index;
    for (index = 0; index < num_input; ++ index)
    {
        test_yolo_results(&net, input_gpath_array[index], thresh, results_array, index, verbose, quiet);
    }
}
