#define PATH_SEP "/"

#include <stdexcept>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>
#include <cmath>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <opencv/highgui.h>

#include "network.h"
#include "yolo.h"

using namespace std;

struct YOLOClass
{
public:
    YOLOClass(bool verbose, bool quiet)
    {
        if( ! quiet )
        {
            #ifdef _OPENMP
                    cout << "[pyrf c++] --- RUNNING PYRF DETECTOR IN PARALLEL ---" << endl;
            #else
                    cout << "[pyrf c++] --- RUNNING PYRF DETECTOR IN SERIAL ---" << endl;
            #endif
        }
    }

    network load(char *cfgfile, char *weightfile, bool verbose, bool quiet)
    {
        // Init network
        network net = parse_network_cfg(cfgfile);
        if(weightfile){
            load_weights(&net, weightfile);
        }
        return net;
    }

    // Run detector
    int detect(network net, char *filename, float thresh)
    {
        detection_layer l = net.layers[net.n-1];
        set_batch_network(&net, 1);
        srand(2222222);
        clock_t time;
        char buff[256];
        char *input = buff;
        int j;
        float nms=.5;
        box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
        float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
        for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
            strncpy(input, filename, 256);
            image im = load_image_color(input,0,0);
            image sized = resize_image(im, net.w, net.h);
            float *X = sized.data;

            time=clock();
            float *predictions = network_predict(net, X);
            if(net.verbose)
            {
                printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
            }

            convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
            if(nms)
            {
                do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
            }

            free_image(im);
            free_image(sized);
    }
};
