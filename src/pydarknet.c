
#include "pydarknet.h"
#include "network.h"
#include "parser.h"
#include "yolo.h"
#include "utils.h"

typedef unsigned char uint8;

#define PYTHON_DARKNET extern DARKNET_DETECTOR_EXPORT

PYTHON_DARKNET network* init(char *config_filepath, char *weight_filepath, int verbose, int quiet)
{
    if ( quiet == 0)
    {
        #ifdef GPU
            printf("[pydarknet c] Using GPU (CUDA)\n");
        #else
            printf("[pydarknet c] Using CPU\n");
        #endif

    }

    printf("[pydarknet c] Building model...");
    fflush(stdout);
    network net = parse_network_cfg(config_filepath, verbose);

    network* holder = malloc(sizeof *holder);
    if (holder == NULL) {
        /* failed to allocate, handle error here */
    } else {
        memcpy(holder, &net, sizeof *holder);
    }

    printf("Done!\n[pydarknet c] ");
    fflush(stdout);
    if(weight_filepath){
        load_weights(&net, weight_filepath);
    }

    return holder;
}

PYTHON_DARKNET void detect(network *net, char **input_gpath_array,
                           int num_input, float thresh, float* results_array,
                           int verbose, int quiet)
{
    printf("\n[pydarknet c] Performing inference on %d images\n", num_input);
    int index;
    for (index = 0; index < num_input; ++ index)
    {
        test_yolo_results(net, input_gpath_array[index], thresh, results_array, index, verbose, quiet);
    }
}
