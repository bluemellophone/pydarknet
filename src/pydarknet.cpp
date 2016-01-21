#include <iostream>
#include <fstream>
#include <vector>

#include "pydarknet.h"

using namespace std;

typedef unsigned char uint8;

#ifdef __cplusplus
extern "C"
{
#endif

#include "network.h"
#include "parser.h"
#include "yolo.h"

#define PYTHON_DARKNET extern DARKNET_DETECTOR_EXPORT

PYTHON_DARKNET void detect(char *config_filepath, char *weight_filepath, char **input_gpath_array,
                           int num_input, float thresh, float* results_array,
                           bool verbose, bool quiet)
{
    if (! quiet)
    {
        #ifndef GPU
            printf("Using CPU\n");
        #else
            printf("Using GPU (CUDA)\n");
        #endif

    }
    network net = parse_network_cfg(config_filepath, verbose);
    if(weight_filepath){
        load_weights(&net, weight_filepath);
    }

    for (int index = 0; index < num_input; ++ index)
    {
        test_yolo_results(&net, input_gpath_array[index], thresh, results_array, index, (int) verbose, (int) quiet);
    }
}
#ifdef __cplusplus
}
#endif
