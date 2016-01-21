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
from pydarknet.pydarknet_helpers import (_load_c_shared_library, _cast_list_to_c, ensure_bytes_strings)


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
# NP_ARRAY_FLOAT = np.ctypeslib.ndpointer(dtype=NP_FLOAT32,     ndim=2, flags=NP_FLAGS)
NP_ARRAY_FLOAT = np.ctypeslib.ndpointer(dtype=NP_FLOAT32,     ndim=1, flags=NP_FLAGS)
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

METHODS['detect'] = ([
    C_CHAR,          # config_filepath
    C_CHAR,          # weight_filepath
    C_ARRAY_CHAR,    # input_gpath_array
    C_INT,           # num_input
    C_FLOAT,         # sensitivity
    NP_ARRAY_FLOAT,  # results_array
    C_BOOL,          # verbose
    C_BOOL,          # quiet
], None)

CLASS_LIST = [
    'elephant_savanna',
    'giraffe_reticulated',
    'giraffe_masai',
    'zebra_grevys',
    'zebra_plains'
]
SIDES = 7
BOXES = 2
PROB_RESULT_LENGTH = SIDES * SIDES * BOXES * len(CLASS_LIST)
BBOX_RESULT_LENGTH = SIDES * SIDES * BOXES * 4
RESULT_LENGTH = PROB_RESULT_LENGTH + BBOX_RESULT_LENGTH

#=================================
# Load Dynamic Library
#=================================
DARKNET_CLIB = _load_c_shared_library(METHODS)


#=================================
# Darknet YOLO Detector
#=================================
class Darknet_YOLO_Detector(object):

    def __init__(dark, verbose=VERBOSE_DARK, quiet=QUIET_DARK):
        '''
            Create the C object for the PyDarknet YOLO detector.

            Args:
                verbose (bool, optional): verbose flag; defaults to --verbdark flag

            Returns:
                detector (object): the Darknet YOLO Detector object
        '''
        dark.verbose = verbose
        dark.quiet = quiet
        if dark.verbose and not dark.quiet:
            print('[pydarknet py] New Darknet_YOLO Object Created')

    def detect(dark, config_filepath, weight_filepath, input_gpath_list, **kwargs):
        '''
            Run detection with a given loaded forest on a list of images

            Args:
                config_filepath (str): the network definition for YOLO to use
                weight_filepath (str): the network weights for YOLO to use
                input_gpath_list (list of str): the list of image paths that you want
                    to test

            Kwargs:
                sensitivity (float, optional): the sensitivity of the detector, which
                    accepts a value between 0.0 and 1.0; defaults to 0.0
                batch_size (int, optional): the number of images to test at a single
                    time in paralell (if None, the number of CPUs is used); defaults to None
                verbose (bool, optional): verbose flag; defaults to object's verbose or
                    selectively enabled for this function

            Yields:
                (str, (list of dict)): tuple of the input image path and a list
                    of dictionaries specifying the detected bounding boxes

                    The dictionaries returned by this function are of the form:
                        xtl (int): the top left x position of the bounding box
                        ytl (int): the top left y position of the bounding box
                        width (int): the width of the bounding box
                        height (int): the hiehgt of the bounding box
                        class (str): the most probably class detected by the network
                        confidence (float): the confidence that this bounding box is of
                            the class specified by the trees used during testing

        '''
        # Default values
        params = odict([
            ('batch_size',    None),
            ('sensitivity',   0.2),
            ('results_array', None),  # This value always gets overwritten
            ('verbose',       dark.verbose),
            ('quiet',         dark.quiet),
        ])
        params.update(kwargs)

        # Try to determine the parallel processing batch size
        if params['batch_size'] is None:
            try:
                cpu_count = multiprocessing.cpu_count()
                if not params['quiet']:
                    print('[pydarknet py] Detecting with %d CPUs' % (cpu_count, ))
                # params['batch_size'] = cpu_count
                params['batch_size'] = 128
            except:
                params['batch_size'] = 128

        # Data integrity
        assert params['sensitivity'] >= 0 and params['sensitivity'] <= 1.0, \
            'Threshold must be in the range [0, 1].'

        # Run training algorithm
        batch_size = params['batch_size']
        del params['batch_size']  # Remove this value from params
        batch_num = int(len(input_gpath_list) / batch_size) + 1
        # Detect for each batch
        for batch in ut.ProgressIter(range(batch_num), lbl='[pydarknet py]', freq=1, invert_rate=True):
            begin = time.time()
            start = batch * batch_size
            end   = start + batch_size
            if end > len(input_gpath_list):
                end = len(input_gpath_list)
            input_gpath_list_        = input_gpath_list[start:end]
            num_images = len(input_gpath_list_)
            # Final sanity check
            params['results_array'] = np.empty(num_images * RESULT_LENGTH, dtype=C_FLOAT)
            # Make the params_list
            params_list = [
                config_filepath,
                weight_filepath,
                _cast_list_to_c(ensure_bytes_strings(input_gpath_list_), C_CHAR),
                num_images,
            ] + list(params.values())
            DARKNET_CLIB.detect(*params_list)
            results_list = params['results_array']
            conclude = time.time()
            results_list = results_list.reshape( (num_images, -1) )
            if not params['quiet']:
                print('[pydarknet py] Took %r seconds to compute %d images' % (conclude - begin, num_images, ))
            for input_gpath, result_list in zip(input_gpath_list_, results_list):
                probs_list, bbox_list = np.split(result_list, [PROB_RESULT_LENGTH])
                assert probs_list.shape[0] == PROB_RESULT_LENGTH and bbox_list.shape[0] == BBOX_RESULT_LENGTH
                probs_list = probs_list.reshape( (-1, len(CLASS_LIST)) )
                bbox_list = bbox_list.reshape( (-1, 4) )

                result_list_ = []
                for prob_list, bbox in zip(probs_list, bbox_list):
                    class_index = np.argmax(prob_list)
                    class_label = CLASS_LIST[class_index]
                    class_confidence = prob_list[class_index]
                    if class_confidence < params['sensitivity']:
                        continue
                    result_dict = {
                        'xtl'        : int(np.around(bbox[0])),
                        'ytl'        : int(np.around(bbox[1])),
                        'width'      : int(np.around(bbox[2])),
                        'height'     : int(np.around(bbox[3])),
                        'class'      : class_label,
                        'confidence' : class_confidence,
                    }
                    result_list_.append(result_dict)

                yield (input_gpath, result_list_)
            params['results_array'] = None

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
