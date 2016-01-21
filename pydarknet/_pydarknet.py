from __future__ import absolute_import, division, print_function
# Standard
from collections import OrderedDict as odict
import multiprocessing
import ctypes as C
from six.moves import zip, range
# Scientific
import utool as ut
import numpy as np
import time
from pydarknet.pydarknet_helpers import (_load_c_shared_library, _cast_list_to_c, ensure_bytes_strings, _extract_np_array)


VERBOSE_DARK = ut.get_argflag('--verbdark') or ut.VERBOSE
QUIET_DARK   = ut.get_argflag('--quietdark') or ut.QUIET


#============================
# CTypes Interface Data Types
#============================
'''
    Bindings for C Variable Types
'''
NP_FLAGS       = 'aligned, c_contiguous, writeable'
# Primatives
C_OBJ          = C.c_void_p
C_BYTE         = C.c_char
C_CHAR         = C.c_char_p
C_INT          = C.c_int
C_BOOL         = C.c_bool
C_FLOAT        = C.c_float
NP_INT8        = np.uint8
NP_FLOAT32     = np.float32
# Arrays
C_ARRAY_CHAR   = C.POINTER(C_CHAR)
C_ARRAY_FLOAT  = C.POINTER(C_FLOAT)
NP_ARRAY_INT   = np.ctypeslib.ndpointer(dtype=C_INT,          ndim=1, flags=NP_FLAGS)
NP_ARRAY_FLOAT = np.ctypeslib.ndpointer(dtype=NP_FLOAT32,     ndim=2, flags=NP_FLAGS)
RESULTS_ARRAY  = np.ctypeslib.ndpointer(dtype=NP_ARRAY_FLOAT, ndim=1, flags=NP_FLAGS)


#=================================
# Method Parameter Types
#=================================
'''
IMPORTANT:
    For functions that return void, use Python None as the return value.
    For functions that take no parameters, use the Python empty list [].
'''

METHODS = {}
METHODS['init'] = ([
    C_BOOL,          # verbose
    C_BOOL,          # quiet
], C_OBJ)

METHODS['detect'] = ([
    C_OBJ,           # detector
    C_OBJ,           # forest
    C_ARRAY_CHAR,    # input_gpath_array
    C_INT,           # _input_gpath_num
    C_ARRAY_CHAR,    # output_gpath_array
    C_ARRAY_CHAR,    # output_scale_gpath_array
    C_INT,           # mode
    C_FLOAT,         # sensitivity
    C_ARRAY_FLOAT,   # scale_array
    C_INT,           # _scale_num
    C_INT,           # nms_min_area_contour
    C_FLOAT,         # nms_min_area_overlap
    RESULTS_ARRAY,   # results_val_array
    NP_ARRAY_INT,    # results_len_array
    C_INT,           # RESULT_LENGTH
    C_BOOL,          # serial
    C_BOOL,          # verbose
    C_BOOL,          # quiet
], None)
RESULT_LENGTH = 8

#=================================
# Load Dynamic Library
#=================================
DARKNET_CLIB = _load_c_shared_library(METHODS)


#=================================
# Darknet YOLO Detector
#=================================
class Random_Forest_Detector(object):

    def __init__(dark, verbose=VERBOSE_DARK, quiet=QUIET_DARK):
        '''
            Create the C object for the PyDarknet detector.

            Args:
                verbose (bool, optional): verbose flag; defaults to --verbdark flag

            Returns:
                detector (object): the Darknet YOLO Detector object
        '''
        dark.verbose = verbose
        dark.quiet = quiet
        if dark.verbose and not dark.quiet:
            print('[pydarknet py] New Darknet_YOLO Object Created')
        dark.detector_c_obj = DARKNET_CLIB.init(dark.verbose, dark.quiet)

    def detect(dark, forest, input_gpath_list, **kwargs):
        '''
            Run detection with a given loaded forest on a list of images

            Args:
                forest (object): the forest obejct that you want to use during
                    detection
                input_gpath_list (list of str): the list of image paths that you want
                    to test

            Kwargs:
                output_gpath_list (list of str, optional): the paralell list of output
                    image paths for detection debugging or results; defaults to None

                    When this list is None no images are outputted for any test
                    images, whereas the list can be a parallel list where some values
                    are strings and others are None
                output_scale_gpath_list (list of str, optional): the paralell list of output
                    scale image paths for detection debugging or results; defaults
                    to None

                    When this list is None no images are outputted for any test
                    images, whereas the list can be a parallel list where some values
                    are strings and others are None
                mode (int, optional): the mode that the detector outputs; detaults to 0
                    0 - Hough Voting - the output is a Hough image that predicts the
                        locations of the obejct centeroids
                    0 - Classification Map - the output is a classification probability
                        map across the entire image where no regression information
                        is utilized
                sensitivity (float, optional): the sensitivity of the detector;

                        mode = 0 - defaults to 128.0
                        mode = 1 - defaults to 255.0

                scale_list (list of float, optional): the list of floats that specifies the scales
                    to try during testing;
                    defaults to [1.0, 0.80, 0.65, 0.50, 0.40, 0.30, 0.20, 0.10]

                        scale > 1.0 - Upscale the image
                        scale = 1.0 - Original image size
                        scale < 1.0 - Downscale the image

                    The list of scales highly impacts the performance of the detector and
                    should be carefully chosen

                    The scales are applied to BOTH the width and the height of the image
                    in order to scale the image and an interpolation of OpenCV's
                    CV_INTER_LANCZOS4 is used
                batch_size (int, optional): the number of images to test at a single
                    time in paralell (if None, the number of CPUs is used); defaults to None
                nms_min_area_contour (int, optional): the minimum size of a centroid
                    candidate region; defaults to 300
                nms_min_area_overlap (float, optional, DEPRICATED): the allowable overlap in
                    bounding box predictions; defaults to 0.75
                serial (bool, optional): flag to signify if to run detection in serial;

                        len(input_gpath_list) >= batch_size - defaults to False
                        len(input_gpath_list) <  batch_size - defaults to False

                verbose (bool, optional): verbose flag; defaults to object's verbose or
                    selectively enabled for this function

            Yields:
                (str, (list of dict)): tuple of the input image path and a list
                    of dictionaries specifying the detected bounding boxes

                    The dictionaries returned by this function are of the form:
                        centerx (int): the x position of the object's centroid

                            Note that the center of the bounding box and the location of
                            the object's centroid can be different
                        centery (int): the y position of the obejct's centroid

                            Note that the center of the bounding box and the location of
                            the object's centroid can be different
                        xtl (int): the top left x position of the bounding box
                        ytl (int): the top left y position of the bounding box
                        width (int): the width of the bounding box
                        height (int): the hiehgt of the bounding box
                        confidence (float): the confidence that this bounding box is of
                            the class specified by the trees used during testing
                        suppressed (bool, DEPRICATED): the flag of if this bounding
                            box has been marked to be suppressed by the detection
                            algorithm

        '''
        # Default values
        params = odict([
            ('output_gpath_list',            None),
            ('output_scale_gpath_list',      None),
            ('mode',                         0),
            ('sensitivity',                  None),
            ('scale_list',                   [1.0, 0.80, 0.65, 0.50, 0.40, 0.30, 0.20, 0.10]),
            ('_scale_num',                   None),  # This value always gets overwritten
            ('batch_size',                   None),
            ('nms_min_area_contour',         100),
            ('nms_min_area_overlap',         0.75),
            ('results_val_array',            None),  # This value always gets overwritten
            ('results_len_array',            None),  # This value always gets overwritten
            ('RESULT_LENGTH',                None),  # This value always gets overwritten
            ('serial',                       False),
            ('verbose',                      dark.verbose),
            ('quiet',                        dark.quiet),
        ])
        params.update(kwargs)
        params['RESULT_LENGTH'] = RESULT_LENGTH
        output_gpath_list = params['output_gpath_list']
        output_scale_gpath_list = params['output_scale_gpath_list']
        # We no longer want these parameters in params
        del params['output_gpath_list']
        del params['output_scale_gpath_list']

        if params['sensitivity'] is None:
            assert params['mode'] in [0, 1], 'Invalid mode provided'
            if params['mode'] == 0:
                params['sensitivity'] = 128.0
            elif params['mode'] == 1:
                params['sensitivity'] = 255.0

        # Try to determine the parallel processing batch size
        if params['batch_size'] is None:
            try:
                cpu_count = multiprocessing.cpu_count()
                if not params['quiet']:
                    print('[pydarknet py] Detecting with %d CPUs' % (cpu_count, ))
                params['batch_size'] = cpu_count
            except:
                params['batch_size'] = 8

        # To eleminate downtime, add 1 to batch_size
        # params['batch_size'] +=

        # Data integrity
        assert params['mode'] >= 0, \
            'Detection mode must be non-negative'
        assert 0.0 <= params['sensitivity'], \
            'Sensitivity must be non-negative'
        assert len(params['scale_list']) > 0 , \
            'The scale list cannot be empty'
        assert all( [ scale > 0.0 for scale in params['scale_list'] ]), \
            'All scales must be positive'
        assert params['batch_size'] > 0, \
            'Batch size must be positive'
        assert params['nms_min_area_contour'] > 0, \
            'Non-maximum suppression minimum contour area cannot be negative'
        assert 0.0 <= params['nms_min_area_overlap'] and params['nms_min_area_overlap'] <= 1.0, \
            'Non-maximum supression minimum area overlap percentage must be between 0 and 1 (inclusive)'

        # Convert optional parameters to C-valid default options
        if output_gpath_list is None:
            output_gpath_list = [''] * len(input_gpath_list)
        elif output_gpath_list is not None:
            assert len(output_gpath_list) == len(input_gpath_list), \
                'Output image path list is invalid or is not the same length as the input list'
            for index in range(len(output_gpath_list)):
                if output_gpath_list[index] is None:
                    output_gpath_list[index] = ''
        output_gpath_list = _cast_list_to_c(ensure_bytes_strings(output_gpath_list), C_CHAR)

        if output_scale_gpath_list is None:
            output_scale_gpath_list = [''] * len(input_gpath_list)
        elif output_scale_gpath_list is not None:
            assert len(output_scale_gpath_list) == len(input_gpath_list), \
                'Output scale image path list is invalid or is not the same length as the input list'
            for index in range(len(output_scale_gpath_list)):
                if output_scale_gpath_list[index] is None:
                    output_scale_gpath_list[index] = ''
        output_scale_gpath_list = _cast_list_to_c(ensure_bytes_strings(output_scale_gpath_list), C_CHAR)

        # Prepare for C
        params['_scale_num'] = len(params['scale_list'])
        params['scale_list'] = _cast_list_to_c(params['scale_list'], C_FLOAT)
        if not params['quiet']:
            print('[pydarknet py] Detecting over %d scales' % (params['_scale_num'], ))

        # Run training algorithm
        batch_size = params['batch_size']
        del params['batch_size']  # Remove this value from params
        batch_num = int(len(input_gpath_list) / batch_size) + 1
        # Detect for each batch
        for batch in ut.ProgressIter(range(batch_num), lbl="[pydarknet py]", freq=1, invert_rate=True):
            begin = time.time()
            start = batch * batch_size
            end   = start + batch_size
            if end > len(input_gpath_list):
                end = len(input_gpath_list)
            input_gpath_list_        = input_gpath_list[start:end]
            output_gpath_list_       = output_gpath_list[start:end]
            output_scale_gpath_list_ = output_scale_gpath_list[start:end]
            num_images = len(input_gpath_list_)
            # Set image detection to be run in serial if less than half a batch to run
            if num_images < min(batch_size / 2, 8):
                params['serial'] = True
            # Final sanity check
            assert len(input_gpath_list_) == len(output_gpath_list_) and len(input_gpath_list_) == len(output_scale_gpath_list_)
            params['results_val_array'] = np.empty(num_images, dtype=NP_ARRAY_FLOAT)
            params['results_len_array'] = np.empty(num_images, dtype=C_INT)
            # Make the params_list
            params_list = [
                forest,
                _cast_list_to_c(ensure_bytes_strings(input_gpath_list_), C_CHAR),
                num_images,
                _cast_list_to_c(ensure_bytes_strings(output_gpath_list_), C_CHAR),
                _cast_list_to_c(ensure_bytes_strings(output_scale_gpath_list_), C_CHAR)
            ] + list(params.values())
            DARKNET_CLIB.detect(dark.detector_c_obj, *params_list)
            results_list = _extract_np_array(params['results_len_array'], params['results_val_array'], NP_ARRAY_FLOAT, NP_FLOAT32, RESULT_LENGTH)
            conclude = time.time()
            if not params['quiet']:
                print('[pydarknet py] Took %r seconds to compute %d images' % (conclude - begin, num_images, ))
            for input_gpath, result_list in zip(input_gpath_list_, results_list):
                if params['mode'] == 0:
                    result_list_ = []
                    for result in result_list:
                        # Unpack result into a nice Python dictionary and return
                        temp = {}
                        temp['centerx']    = int(result[0])
                        temp['centery']    = int(result[1])
                        temp['xtl']        = int(result[2])
                        temp['ytl']        = int(result[3])
                        temp['width']      = int(result[4])
                        temp['height']     = int(result[5])
                        temp['confidence'] = float(np.round(result[6], decimals=4))
                        temp['suppressed'] = int(result[7]) == 1
                        result_list_.append(temp)
                    yield (input_gpath, result_list_)
                else:
                    yield (input_gpath, None)
            params['results_val_array'] = None
            params['results_len_array'] = None

    # Pickle functions
    def dump(dark, file):
        '''
            UNIMPLEMENTED

            Args:
                file (object)

            Returns:
                None
        '''
        pass

    def dumps(dark):
        '''
            UNIMPLEMENTED

            Returns:
                string
        '''
        pass

    def load(dark, file):
        '''
            UNIMPLEMENTED

            Args:
                file (object)

            Returns:
                detector (object)
        '''
        pass

    def loads(dark, string):
        '''
            UNIMPLEMENTED

            Args:
                string (str)

            Returns:
                detector (object)
        '''
        pass
