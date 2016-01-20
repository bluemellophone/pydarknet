from os.path import isfile, join, abspath
import numpy as np
from six.moves import cPickle as pickle

name = 'yolo.5.weights'
path = '.'
path = abspath(path)

layer = 0
layer_list = []
while True:
    filename = '%s.%d.raw' % (name, layer, )
    filepath = join(path, filename)
    if not isfile(filepath):
        break
    raw = np.fromfile(filepath, dtype=np.int32)

    major = int(raw[0])
    minor = int(raw[1])
    revision = int(raw[2])
    seen = int(raw[3])

    assert major == 0
    assert minor == 1
    assert revision == 0

    raw = raw[4:]
    type_ = int(raw[0])

    if type_ == 0:
        n, c, size = raw[1:4]
        num = n * c * size * size
        expected = num + n
        vals = (layer, n, c, size, size, n, expected, )
        print('[%3d] CONV (%d x %d x %d x %d) + %d = %d' % vals)

        params = raw[4:].view(np.float32)
        try:
            assert expected == len(params)
        except AssertionError:
            print('      ... BATCH NORM')
            raise NotImplementedError

        b = params[:n]
        w = params[n:]

        vals = (layer, b[0], b[-1], w[0], w[-1], )
        print('[%3d]     Check: %0.011f, %0.011f, %0.011f, %0.011f' % vals)

        b_shape = (n, )
        w_shape = (n, c, size, size)
        b = np.reshape(b, b_shape)
        w = np.reshape(w, w_shape)
        # print('     w.shape = %r' % (w.shape, ))
        # print('     b.shape = %r' % (b.shape, ))
        layer_list.append(w)
        layer_list.append(b)
    elif type_ == 2:
        outputs, inputs = raw[1:3]
        expected = outputs * inputs + outputs
        vals = (layer, outputs, inputs, outputs, expected, )
        print('[%3d] DENSE (%d x %d) + %d = %d' % vals)

        params = raw[3:].view(np.float32)
        assert expected == len(params)

        b = params[:outputs]
        w = params[outputs:]

        vals = (layer, b[0], b[-1], w[0], w[-1], )
        print('[%3d]     Check: %0.011f, %0.011f, %0.011f, %0.011f' % vals)

        b_shape = (outputs, )
        w_shape = (inputs, outputs)
        b = np.reshape(b, b_shape)
        w = np.reshape(w, w_shape)
        # print('     w.shape = %r' % (w.shape, ))
        # print('     b.shape = %r' % (b.shape, ))
        layer_list.append(w)
        layer_list.append(b)
    elif type == 12:
        # size, c, n, out_w, out_h, outputs = raw[1:7]
        # expected = size + outputs
        # vals = (layer, size, size, c, n, out_w, out_h, outputs, expected, )
        # print('[%3d] LOCAL (%d x %d x %d x %d x (%d x %d)) + %d = %d' % vals)

        # params = raw[7:]
        # assert expected == len(params)
        raise NotImplementedError
    else:
        vals = (layer, type_)
        print('[%3d] SKIPPED: %d' % vals)

    layer += 1

# for layer in layer_list:
#     print(layer.shape)
# print(len(layer_list))

filename = '%s.pickle' % (name, )
filepath = join(path, filename)
pickle.dump(layer_list, open(filepath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
