
void train_yolo_custom(network *net, char *train_images, char *backup_directory, char *weight_filepath, int verbose, int quiet);
void test_yolo_results(network *net, char *filename, float sensitivity, float* results, int result_index, int verbose, int quiet);
