#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from os.path import join, abspath
from pydarknet import Darknet_YOLO_Detector
import utool as ut


def test_pydarknet():
    r"""
    CommandLine:
        python -m test_pydarknet --test-test_pydarknet

    Example:
        >>> # ENABLE_DOCTEST
        >>> from test_pydarknet import *  # NOQA
        >>> result = test_pydarknet()
        >>> print(result)
    """

    dark = Darknet_YOLO_Detector()
    config_filepath = abspath(join('cfg', 'yolo.cfg'))
    weight_filepath = abspath(join('_test', 'yolo.5.weights'))
    input_gpath_list = [
        abspath(join('_test', 'test_%05d.jpg' % (i, )))
        for i in range(1, 76)
    ]
    input_gpath_list = input_gpath_list[:5]
    results = dark.detect(config_filepath, weight_filepath, input_gpath_list)
    print(list(results))
    return locals()


if __name__ == '__main__':
    test_locals = ut.run_test(test_pydarknet)
    exec(ut.execstr_dict(test_locals, 'test_locals'))
    exec(ut.ipython_execstr())
