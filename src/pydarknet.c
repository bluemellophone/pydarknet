
#include "pydarknet.h"
#include "network.h"
#include "parser.h"
#include "yolo.h"
#include "utils.h"

typedef unsigned char uint8;

#define PYTHON_DARKNET extern DARKNET_DETECTOR_EXPORT

PYTHON_DARKNET float detect(char *config_filepath, char *weight_filepath, char **input_gpath_array,
                           int num_input, float thresh, float* results_array,
                           int verbose, int quiet)
{
    if ( quiet == 0)
    {
        #ifdef GPU
            printf("Using GPU (CUDA)\n");
        #else
            printf("Using CPU\n");
        #endif

    }

    clock_t time = clock();
    printf("[pydarknet c] Building model...");
    fflush(stdout);
    network net = parse_network_cfg(config_filepath, verbose);
    printf("Done!\n[pydarknet c] ");
    fflush(stdout);
    if(weight_filepath){
        load_weights(&net, weight_filepath);
    }
    float load_time = sec(clock() - time);

    printf("[pydarknet c] Performing inference on %d images\n", num_input);
    int index;
    for (index = 0; index < num_input; ++ index)
    {
        test_yolo_results(&net, input_gpath_array[index], thresh, results_array, index, verbose, quiet);
    }

    return load_time;
}
