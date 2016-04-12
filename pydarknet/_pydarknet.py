from __future__ import absolute_import, division, print_function
# Standard
from collections import OrderedDict as odict
# import multiprocessing
import ctypes as C
from six.moves import zip, range
# Scientific
from os.path import abspath, basename, join, exists
import utool as ut
import numpy as np
import time
from pydarknet.pydarknet_helpers import (_load_c_shared_library, _cast_list_to_c, ensure_bytes_strings)


VERBOSE_DARK = ut.get_argflag('--verbdark') or ut.VERBOSE
QUIET_DARK   = ut.get_argflag('--quietdark') or ut.QUIET


DEFAULT_CONFIG_URL          = 'https://lev.cs.rpi.edu/public/models/detect.yolo.12.cfg'
DEFAULT_WEIGHTS_URL         = 'https://lev.cs.rpi.edu/public/models/detect.yolo.12.weights'
OLD_DEFAULT_CONFIG_URL      = 'https://lev.cs.rpi.edu/public/models/detect.yolo.5.cfg'
OLD_DEFAULT_WEIGHTS_URL     = 'https://lev.cs.rpi.edu/public/models/detect.yolo.5.weights'
DEFAULT_CONFIG_TEMPLATE_URL = 'https://lev.cs.rpi.edu/public/models/detect.yolo.template.cfg'
DEFAULT_PRETRAINED_URL      = 'https://lev.cs.rpi.edu/public/models/detect.yolo.pretrained.weights'


#============================
# CTypes Interface Data Types
#============================
"""
    Bindings for C Variable Types
"""
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
"""
IMPORTANT:
    For functions that return void, use Python None as the return value.
    For functions that take no parameters, use the Python empty list [].
"""

METHODS = {}

METHODS['init'] = ([
    C_CHAR,          # config_filepath
    C_CHAR,          # weight_filepath
    C_INT,           # verbose
    C_INT,           # quiet
], C_OBJ)

METHODS['train'] = ([
    C_OBJ,           # network
    C_CHAR,          # train_image_manifest
    C_CHAR,          # weight_path
    C_INT,           # num_input
    C_CHAR,          # final_weight_filepath
    C_INT,           # verbose
    C_INT,           # quiet
], None)

METHODS['detect'] = ([
    C_OBJ,           # network
    C_ARRAY_CHAR,    # input_gpath_array
    C_INT,           # num_input
    C_FLOAT,         # sensitivity
    C_INT,           # grid
    NP_ARRAY_FLOAT,  # results_array
    C_INT,           # verbose
    C_INT,           # quiet
], None)

DEFAULT_CLASS = 'UNKNOWN'
CLASS_LIST = [
    'lion',
    'zebra_plains',
    'hippopotamus',
    'antelope',
    'elephant_savanna',
    'giraffe_reticulated',
    'zebra_grevys',
    'giraffe_masai',
    'unspecified_animal',
    'car',
    'bird',
    'building',
]
OLD_CLASS_LIST = [
    'elephant_savanna',
    'giraffe_reticulated',
    'giraffe_masai',
    'zebra_grevys',
    'zebra_plains',
]
SIDES = 7
BOXES = 2
GRID = 1
PROB_RESULT_LENGTH = None
BBOX_RESULT_LENGTH = None
RESULT_LENGTH = None


def _update_globals(grid=GRID, class_list=CLASS_LIST):
    global PROB_RESULT_LENGTH, BBOX_RESULT_LENGTH, RESULT_LENGTH
    PROB_RESULT_LENGTH = grid * SIDES * SIDES * BOXES * len(class_list)
    BBOX_RESULT_LENGTH = grid * SIDES * SIDES * BOXES * 4
    RESULT_LENGTH = PROB_RESULT_LENGTH + BBOX_RESULT_LENGTH


#=================================
# Load Dynamic Library
#=================================
_update_globals()
DARKNET_CLIB = _load_c_shared_library(METHODS)


#=================================
# Darknet YOLO Detector
#=================================
class Darknet_YOLO_Detector(object):

    def __init__(dark, config_filepath=None, weight_filepath=None,
                 verbose=VERBOSE_DARK, quiet=QUIET_DARK):
        """
            Create the C object for the PyDarknet YOLO detector.

            Args:
                verbose (bool, optional): verbose flag; defaults to --verbdark flag

            Returns:
                detector (object): the Darknet YOLO Detector object
        """

        dark.CLASS_LIST = None
        if config_filepath in ['default', 'v2', None]:
            config_filepath = ut.grab_file_url(DEFAULT_CONFIG_URL, appname='pydarknet')
            dark.CLASS_LIST = CLASS_LIST
        elif config_filepath in ['v1', 'old', 'original']:
            dark.CLASS_LIST = OLD_CLASS_LIST
            config_filepath = ut.grab_file_url(OLD_DEFAULT_CONFIG_URL, appname='pydarknet')

        if weight_filepath in ['default', 'v2', None]:
            weight_filepath = ut.grab_file_url(DEFAULT_WEIGHTS_URL, appname='pydarknet')
        elif weight_filepath in ['v1', 'old', 'original']:
            weight_filepath = ut.grab_file_url(OLD_DEFAULT_WEIGHTS_URL, appname='pydarknet')

        dark.verbose = verbose
        dark.quiet = quiet

        dark._load(config_filepath, weight_filepath)

        if dark.verbose and not dark.quiet:
            print('[pydarknet py] New Darknet_YOLO Object Created')

    def _load(dark, config_filepath, weight_filepath):
        begin = time.time()
        params_list = [
            config_filepath,
            weight_filepath,
            dark.verbose,
            dark.quiet,
        ]
        dark.net = DARKNET_CLIB.init(*params_list)
        conclude = time.time()
        if not dark.quiet:
            print('[pydarknet py] Took %r seconds to load' % (conclude - begin, ))

    def _train_setup(dark, voc_path, weight_path):

        class_list = []
        annotations_path = join(voc_path, 'Annotations')
        imagesets_path  = join(voc_path, 'ImageSets')
        jpegimages_path = join(voc_path, 'JPEGImages')
        label_path      = join(voc_path, 'labels')

        ut.delete(label_path)
        ut.ensuredir(label_path)

        def _convert_annotation(image_id):
            import xml.etree.ElementTree as ET

            def _convert(size, box):
                dw = 1. / size[0]
                dh = 1. / size[1]
                x = (box[0] + box[1]) / 2.0
                y = (box[2] + box[3]) / 2.0
                w = box[1] - box[0]
                h = box[3] - box[2]
                x = x * dw
                w = w * dw
                y = y * dh
                h = h * dh
                return (x, y, w, h)

            with open(join(label_path, '%s.txt' % (image_id, )), 'w') as out_file:
                with open(join(annotations_path, '%s.xml' % (image_id, )), 'r') as in_file:
                    tree = ET.parse(in_file)
                    root = tree.getroot()
                    size = root.find('size')
                    w = int(size.find('width').text)
                    h = int(size.find('height').text)

                    for obj in root.iter('object'):
                        if int(obj.find('difficult').text) == 1:
                            continue
                        class_ = obj.find('name').text
                        if class_ not in class_list:
                            class_list.append(class_)
                        class_id = class_list.index(class_)
                        xmlbox = obj.find('bndbox')
                        b = tuple(map(float, [
                            xmlbox.find('xmin').text,
                            xmlbox.find('xmax').text,
                            xmlbox.find('ymin').text,
                            xmlbox.find('ymax').text,
                        ]))
                        bb = _convert((w, h), b)
                        bb_str = ' '.join( [str(_) for _ in bb] )
                        out_file.write('%s %s\n' % (class_id, bb_str))

        num_images = 0
        print('[pydarknet py train] Processing manifest...')
        manifest_filename = join(voc_path, 'manifest.txt')
        with open(manifest_filename, 'w') as manifest:
            # for dataset_name in ['train', 'val', 'test']:
            for dataset_name in ['train', 'val']:
                dataset_filename = join(imagesets_path, 'Main', '%s.txt' % dataset_name)
                with open(dataset_filename, 'r') as dataset:
                    image_id_list = dataset.read().strip().split()

                for image_id in image_id_list:
                    print('[pydarknet py train]     processing: %r' % (image_id, ))
                    image_filepath = abspath(join(jpegimages_path, '%s.jpg' % image_id))
                    if exists(image_filepath):
                        manifest.write('%s\n' % (image_filepath, ))
                        _convert_annotation(image_id)
                        num_images += 1

        print('[pydarknet py train] Processing config and pretrained weights...')
        # Load default config and pretrained weights
        config_filepath = ut.grab_file_url(DEFAULT_CONFIG_TEMPLATE_URL, appname='pydarknet')
        with open(config_filepath, 'r') as config:
            config_template_str = config.read()

        config_filename = basename(config_filepath).replace('.template.', '.%d.' % (len(class_list), ))
        config_filepath = join(weight_path, config_filename)
        with open(config_filepath, 'w') as config:
            replace_list = [
                ('_^_OUTPUT_^_',  SIDES * SIDES * (BOXES * 5 + len(class_list))),
                ('_^_CLASSES_^_', len(class_list)),
                ('_^_SIDES_^_',   SIDES),
                ('_^_BOXES_^_',   BOXES),
            ]
            for needle, replacement in replace_list:
                config_template_str = config_template_str.replace(needle, str(replacement))
            config.write(config_template_str)

        class_filepath = '%s.classes' % (config_filepath, )
        with open(class_filepath, 'w') as class_file:
            for class_ in class_list:
                class_file.write('%s\n' % (class_, ))

        weight_filepath = ut.grab_file_url(DEFAULT_PRETRAINED_URL, appname='pydarknet')
        dark._load(config_filepath, weight_filepath)

        print('class_list = %r' % (class_list, ))
        print('num_images = %r' % (num_images, ))

        return manifest_filename, num_images

    def train(dark, voc_path, weight_path, **kwargs):
        """
            Train a new forest with the given positive chips and negative chips.

            Args:
                train_pos_chip_path_list (list of str): list of positive training chips
                train_neg_chip_path_list (list of str): list of negative training chips
                trees_path (str): string path of where the newly trained trees are to be saved

            Kwargs:
                chips_norm_width (int, optional): Chip normalization width for resizing;
                    the chip is resized to have a width of chips_norm_width and
                    whatever resulting height in order to best match the original
                    aspect ratio; defaults to 128

                    If both chips_norm_width and chips_norm_height are specified,
                    the original aspect ratio of the chip is not respected
                chips_norm_height (int, optional): Chip normalization height for resizing;
                    the chip is resized to have a height of chips_norm_height and
                    whatever resulting width in order to best match the original
                    aspect ratio; defaults to None

                    If both chips_norm_width and chips_norm_height are specified,
                    the original aspect ratio of the chip is not respected
                verbose (bool, optional): verbose flag; defaults to object's verbose or
                    selectively enabled for this function

            Returns:
                None
        """
        # Default values
        params = odict([
            ('weight_filepath', None),  # This value always gets overwritten
            ('verbose',         dark.verbose),
            ('quiet',           dark.quiet),
        ])
        # params.update(kwargs)
        ut.update_existing(params, kwargs)

        # Make the tree path absolute
        weight_path = abspath(weight_path)
        ut.ensuredir(weight_path)

        # Setup training files and folder structures
        manifest_filename, num_images = dark._train_setup(voc_path, weight_path)

        # Run training algorithm
        params_list = [
            dark.net,
            manifest_filename,
            weight_path,
            num_images,
        ] + list(params.values())
        DARKNET_CLIB.train(*params_list)
        weight_filepath = params['weight_filepath']

        if not params['quiet']:
            print('\n\n[pydarknet py] *************************************')
            print('[pydarknet py] Training Completed')
            print('[pydarknet py] Weight file saved to: %s' % (weight_filepath, ))

    def detect(dark, input_gpath_list, **kwargs):
        """
            Run detection with a given loaded forest on a list of images

            Args:
                input_gpath_list (list of str): the list of image paths that you want
                    to test
                config_filepath (str, optional): the network definition for YOLO to use
                weight_filepath (str, optional): the network weights for YOLO to use

            Kwargs:
                sensitivity (float, optional): the sensitivity of the detector, which
                    accepts a value between 0.0 and 1.0; defaults to 0.0
                batch_size (int, optional): the number of images to test at a single
                    time in paralell (if None, the number of CPUs is used); defaults to
                    None
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

        """
        # Default values
        params = odict([
            ('batch_size',    None),
            ('class_list',    dark.CLASS_LIST),
            ('sensitivity',   0.2),
            ('grid',          False),
            ('results_array', None),  # This value always gets overwritten
            ('verbose',       dark.verbose),
            ('quiet',         dark.quiet),
        ])
        # params.update(kwargs)
        ut.update_existing(params, kwargs)
        class_list = params['class_list']
        del params['class_list']  # Remove this value from params

        if params['grid']:
            _update_globals(grid=10, class_list=class_list)
        else:
            _update_globals(grid=1, class_list=class_list)

        # Try to determine the parallel processing batch size
        if params['batch_size'] is None:
            # try:
            #     cpu_count = multiprocessing.cpu_count()
            #     if not params['quiet']:
            #         print('[pydarknet py] Detecting with %d CPUs' % (cpu_count, ))
            #     params['batch_size'] = cpu_count
            # except:
            #     params['batch_size'] = 128
            params['batch_size'] = 32

        params['verbose'] = int(params['verbose'])
        params['quiet'] = int(params['quiet'])

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
                dark.net,
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
                probs_list = probs_list.reshape( (-1, len(class_list)) )
                bbox_list = bbox_list.reshape( (-1, 4) )

                result_list_ = []
                for prob_list, bbox in zip(probs_list, bbox_list):
                    class_index = np.argmax(prob_list)
                    class_label = class_list[class_index] if len(class_list) > class_index else DEFAULT_CLASS
                    class_confidence = prob_list[class_index]
                    if class_confidence < params['sensitivity']:
                        continue
                    result_dict = {
                        'xtl'        : int(np.around(bbox[0])),
                        'ytl'        : int(np.around(bbox[1])),
                        'width'      : int(np.around(bbox[2])),
                        'height'     : int(np.around(bbox[3])),
                        'class'      : class_label,
                        'confidence' : float(class_confidence),
                    }
                    result_list_.append(result_dict)

                yield (input_gpath, result_list_)
            params['results_array'] = None

    # Pickle functions
    def dump(dark, file):
        """
            UNIMPLEMENTED

            Args:
                file (object)

            Returns:
                None
        """
        pass

    def dumps(dark):
        """
            UNIMPLEMENTED

            Returns:
                string
        """
        pass

    def load(dark, file):
        """
            UNIMPLEMENTED

            Args:
                file (object)

            Returns:
                detector (object)
        """
        pass

    def loads(dark, string):
        """
            UNIMPLEMENTED

            Args:
                string (str)

            Returns:
                detector (object)
        """
        pass


def test_pydarknet():
    r"""

    CommandLine:
        python -m pydarknet._pydarknet --exec-test_pydarknet --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pydarknet._pydarknet import *  # NOQA
        >>> result = test_pydarknet()
        >>> print(result)
    """
    # import ibeis
    # from ibeis.other.detectfuncs import export_to_xml

    dark = Darknet_YOLO_Detector()

    input_gpath_list = [
        abspath(join('_test', 'test_%05d.jpg' % (i, )))
        for i in range(1, 76)
    ]
    input_gpath_list = input_gpath_list[:5]

    results_list = dark.detect(input_gpath_list)
    for filename, result_list in results_list:
        print(filename)
        for result in result_list:
            print('    Found: %r' % (result, ))

    # ibs database from mtest
    voc_path = '/media/extend/jason/Dataset/'
    weight_path = '/media/extend/jason/weights'
    ut.ensuredir(weight_path)
    dark.train(voc_path, weight_path)

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m pydarknet._pydarknet
        python -m pydarknet._pydarknet --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
