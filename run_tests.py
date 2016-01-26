#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
from os.path import join, abspath
from pydarknet import Darknet_YOLO_Detector
import utool as ut
import ibeis
from ibeis.ibsfuncs import export_to_xml


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

    input_gpath_list = [
        abspath(join('_test', 'test_%05d.jpg' % (i, )))
        for i in range(1, 76)
    ]
    # input_gpath_list = input_gpath_list[:5]

    results_list = dark.detect(input_gpath_list)
    for filename, result_list in results_list:
        print(filename)
        for result in result_list:
            print('    Found: %r' % (result, ))

    # ibs database from mtest
    ibs = ibeis.opendb(db='PZ_MTEST')
    export_to_xml(ibs)
    voc_path = abspath(join(ibs._ibsdb, 'LearningData'))
    weight_path = abspath(join(ibs._ibsdb, 'weights'))
    ut.ensuredir(weight_path)
    dark.train(voc_path, weight_path)

    return locals()


if __name__ == '__main__':
    test_locals = ut.run_test(test_pydarknet)
    exec(ut.execstr_dict(test_locals, 'test_locals'))
    exec(ut.ipython_execstr())
