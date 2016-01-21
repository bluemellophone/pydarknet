#============================
# Python Interface
#============================
from __future__ import absolute_import, division, print_function
from os.path import join, realpath, dirname
import numpy as np
import ctypes as C
import sys
import detecttools.ctypes_interface as ctypes_interface


def ensure_bytes_strings(str_list):
    # converts python3 strings into bytes
    if sys.hexversion > 0x03000000:
        return [str_ if not isinstance(str_, str) else bytes(str_, 'utf-8') for str_ in str_list]
    else:
        return str_list


def _cast_list_to_c(py_list, dtype):
    """
    Converts a python list of strings into a c array of strings
    adapted from "http://stackoverflow.com/questions/3494598/passing-a-list-of
    -strings-to-from-python-ctypes-to-c-function-expecting-char"
    Avi's code
    """
    c_arr = (dtype * len(py_list))()
    c_arr[:] = py_list
    return c_arr


def _arrptr_to_np(c_arrptr, shape, arr_t, dtype):
    """
    Casts an array pointer from C to numpy
    Input:
        c_arrpt - an array pointer returned from C
        shape   - shape of that array pointer
        arr_t   - the ctypes datatype of c_arrptr
    Avi's code
    """
    arr_t_size = C.POINTER(C.c_char * dtype().itemsize)           # size of each item
    c_arr = C.cast(c_arrptr.astype(int), arr_t_size)              # cast to ctypes
    np_arr = np.ctypeslib.as_array(c_arr, shape)                  # cast to numpy
    np_arr.dtype = dtype                                          # fix numpy dtype
    np_arr = np.require(np_arr, dtype=dtype, requirements=['O'])  # prevent memory leaks
    return np_arr


def _extract_np_array(size_list, ptr_list, arr_t, arr_dtype,
                        arr_dim):
    """
    size_list - contains the size of each output 2d array
    ptr_list  - an array of pointers to the head of each output 2d
                array (which was allocated in C)
    arr_t     - the C pointer type
    arr_dtype - the numpy array type
    arr_dim   - the number of columns in each output 2d array
    """
    arr_list = [_arrptr_to_np(arr_ptr, (size, arr_dim), arr_t, arr_dtype)
                    for (arr_ptr, size) in zip(ptr_list, size_list)]
    return arr_list


def _load_c_shared_library(METHODS):
    ''' Loads the pydarknet dynamic library and defines its functions '''
    root_dir = realpath(join('..', dirname(__file__)))
    libname = 'pydarknet'
    darknet_clib, def_cfunc = ctypes_interface.load_clib(libname, root_dir)
    # Load and expose methods from lib
    for method in METHODS.keys():
        def_cfunc(METHODS[method][1], method, METHODS[method][0])
    return darknet_clib
