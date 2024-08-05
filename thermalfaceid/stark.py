# Copyright (C) Meridian Innovation Ltd. Hong Kong, 2019 - 2022. All rights reserved.
#
from functools import partial
import numpy as np
import cv2 as cv
from senxor.utils import remap
from senxor.filters import RollingAverageFilter, TrueAverageFilter

# STARK filter parameters
# ===========================
STARK_LMAV_DIFF = {'lm_atype': 'ra', 'lm_ks': (5,5), 'lm_ad': 12}
STARK_SELF_DIFF = {'lm_atype': 'ra', 'lm_ks': (3,3), 'lm_ad': 0}
STARK_SIGMOID = {'alpha': 2.0, 'beta': 2.0}
STARK_SMOOTHSTEP = {'alpha': 0.3, 'beta': 0.0, 'delta': 0.02,
                    'quad': 0.8, 'cube': -0.9}
STARK_CUBE = {'alpha': 0.3, 'beta': 0.0, 'delta': 0.02,
              'quad': 0.8, 'cube': -0.9, 'max': 0.8}

def sigmoid(x, alpha, beta):
    return 1. / (1 + np.exp(-alpha * (x - beta)))

def cube(x, alpha, beta, delta, quad, cube):
    _x = -alpha * (x - beta)
    x2 = _x * _x
    x3 = x2 * _x
    res = quad * x2 + cube * x3 + delta
    res[res > maxim] = maxim
    return res

def smoothstep(x, alpha, beta, delta, quad, cube):
    _x = -alpha * (x - beta)
    x2 = _x * _x
    x3 = x2 * _x
    res = quad * x2 + cube * x3 + delta
    res[res > 0.85] = 0.85
    return res

def set_stark_param(variant=None, sigmoid='sigmoid', module_type='cougar'):
    param = {}
    param['module_type'] = module_type
    #
    if variant not in ['original', 'quick']:
        raise RuntimeError(f'Given STARK variant {variant} not supported.')
    if sigmoid not in ['sigmoid', 'smoothstep', 'cube']:
        raise RuntimeError(f'Given STARK sigmoid {sigmoid} not supported.')
    param['variant'] = variant
    param['sigmoid'] = sigmoid
    if variant == 'original': param.update(STARK_LMAV_DIFF)
    if variant == 'quick': param.update(STARK_SELF_DIFF)
    #
    if sigmoid == 'sigmoid': param.update(STARK_SIGMOID)
    if sigmoid == 'cube': param.update(STARK_CUBE)
    if sigmoid == 'smoothstep': param.update(STARK_SMOOTHSTEP)
    return param


class STARKFilter:

    def __init__(self, param):
        """
        Spatio-Temporal Advanced Rolling Kernel Filter

        Usage:

            frame_filter = STARKFilter()
            ...

            filtered = frame_filter(input_data)
        """
        print('STARK parameters: ', param)
        # local mean can have different kernel size, e.g. 3x3 or 5x5 pixels
        lm_ks = param['lm_ks']
        # local mean can have a temporal average: true or rolling
        lm_atype = param.get('lm_atype', 'ra')
        # local mean temporal average can have various depth 
        lm_ad = param['lm_ad']
        # describe the sigmoid which controls the gain based on the difference
        # between a historical value (of or around) a pixel, and the
        # instantaneous local mean at the pixel
        # alpha and beta and delta are response acceleration, noise barrier
        # and minimum fraction of frame update
        if param.get('sigmoid_type', 'sigmoid') == 'sigmoid':
            alpha = param['alpha']
            beta  = param['beta']
            self.sigmoid = partial(sigmoid, alpha=alpha, beta=beta)
        if param['sigmoid'] in ['smoothstep', 'cube']:
            alpha = param['alpha']
            beta  = param['beta']
            delta = param['delta']
            quad  = param['quad']
            cube  = param['cube']
            self.sigmoid = partial(smoothstep, alpha=alpha, beta=beta, delta=delta,
                                  quad=quad, cube=cube)
        # local mean kernel size
        self.lm_ks = lm_ks
        # temporal average filter on the local mean
        # if we have a temporal average of the local mean, then we use this
        # against the instantaneous mean to predict motion
        if lm_ad > 0:
            if lm_atype == 'ra':
                self.lm_av = RollingAverageFilter(lm_ad)
            if lm_atype == 'ta':
                self.lm_av = TrueAverageFilter(lm_ad)
            self.get_diff = self.diff_lmav
        else:
            self.get_diff = self.diff_self
        # STARK filter output. Init to 0 int, so as to avoid figuring out the shape
        self.av = 0

    def diff_self(self, lm):
        return lm - self.av

    def diff_lmav(self, lm):
        """
        Compute the difference between instantaneous local mean and the historical
        average of the local mean. Then update the historical average of the local mean.
        """
        diff = lm - self.lm_av.av
        return diff

    def update(self, new):
        minmax = new.min(), new.max()
        # the following allows `new` to be 3-channel image (cv.MAT of uint8)
        # such a matrix needs no remapping and thus STARK can be applied to
        # visual camera input as well
        if len(new.shape) == 2:
            new_u8 = remap(new)
        else:
            new_u8 = new
        # smooth input and convert it back to temperature:
        # we must explicitly specify both current and new range
        # we can work with uint for new_lm and lm, but it becomes more difficult to
        # set beta correctly
        new_lm = cv.blur(new_u8, self.lm_ks)
        # remap from uint to temp only if one-channel 3D
        if len(new.shape) == 2:
            new_lm = remap(new_lm, curr_range=(0,255), new_range=minmax, to_uint8=False)
        self.x = self.get_diff(new_lm)
        self.gamma = self.sigmoid(np.abs(self.x))
        self.av += self.gamma * (new - self.av)
        # try to cure the noisy pixels by replacing them with the blurred
        # self.av[self.gamma > 0.9] = 0.5 * (self.av[self.gamma > 0.9] +\
        #                                    new_lm[self.gamma > 0.9])
        self.new_lm = new_lm
        # update the local mean average last
        if self.get_diff == self.diff_lmav: self.lm_av(new_lm)

    def _quick_update(self, new):
        minmax = new.min(), new.max()
        # the following allows `new` to be 3-channel image (cv.MAT of uint8)
        # such a matrix needs no remapping and thus STARK can be applied to
        # visual camera input as well
        if len(new.shape) == 2:
            new_u8 = remap(new)
        else:
            new_u8 = new
        # smooth input and convert it back to temperature:
        # we must explicitly specify both current and new range
        # we can work with uint for new_lm and lm, but it becomes more difficult to
        # set beta correctly
        new_lm = cv.blur(new_u8, self.lm_ks)
        # remap from uint to temp only if one-channel 3D
        if len(new.shape) == 2:
            new_lm = remap(new_lm, curr_range=(0,255), new_range=minmax, to_uint8=False)
        self.x = new_lm - self.av
        # sigmoid on the difference
        # self.gamma = 1. / (1 + np.exp(-self.alpha * (np.abs(x) - self.beta)))
        self.gamma = self.sigmoid(self.x)
        self.av += self.gamma * (new - self.av)

    def __call__(self, new):
        self.update(new)
        return self.av


class Kalman_with_predict():
    """ Kalman implementation. Requires an initial frame i.e., frame0 must be pased as input
        For better results, frame0 can be an average of 2 successive frames or more.
        Else a very random interger is initialized as initial temparature reading.
        Performance is better when used after filter.
        mea_err: the error measured between new frame and previous estimated frame
        err_est: the mean estimated error. can be initialized randomly or using calibration data.
                 will decay over time.
        process_error: the mean of the std per pixel matrix obtained from calibration or a fixed
                       std for all pixels. If std for all pixels is used, then use the
                       mean gain rather than the gain per pixel matrix.
        update_which: Which loss to use to compute the mean_err (mean erro) t iteration t.
                      l1 and l2 are both implemented.
        see: https://github.com/Ugenteraan/Kalman-Filter-Scratch/blob/master/Kalman-Scratch-Implementation.ipynb
    """

    def __init__(self, frame0=None, update_which='l1', smooth_new_frame=False, which_predict='Median', ksize=3,
                 sigma=1, cnn_model=None, use_normalization=1, r_depth=4, err=1.5, gain_scale=0.35,
                 gain_which='stair-case', out_smooth=False, offset=0.1, scale=0.9):

        # should the new input be smooth before Kalman?
        self.smooth_new_frame = smooth_new_frame

        # use l1 or l2 loss
        self.update_which = update_which

        # initial frame keeping or initialize complete random constant integer between 0 and 37 as initial value
        # and rely on python broadcasting to treat it as matrix for gain update
        if frame0 is None:
            self.est = np.random.randint(low=0, high=37)
            self.min_temp = 16
            self.max_temp = 34
        else:
            self.est = frame0
            self.min_temp = frame0.min()
            self.max_temp = frame0.max()

        # initialize method of update for gain
        self.gain_which = gain_which
        if self.gain_which == 'stair-case':
            self.update_which = 'l1'

        # smooth output or raw improved output
        self.out_smooth = out_smooth

        # initialize rolling average depth
        self.r_depth = r_depth

        # initialize pixel err std for sigmoid
        self.pixel_err_std = err

        # initialize gain scale
        self.gain_scale = gain_scale

        #initialize counter
        self.count = 0

        # for poly
        self.offset = offset
        self.scale = scale

        # initialize prediction parameters
        assert which_predict in ['Median', 'Box', 'Gaussian', 'CNN', None]
        self.which_predict = which_predict
        self.ksize = ksize
        self.sigma = sigma

        if self.which_predict == 'CNN':
            self.model = cnn_model
            self.use_normalization = use_normalization

    def to_tensor(self, img):
        if img.ndim == 2:
            return img[np.newaxis, ..., np.newaxis]

    def from_tensor(self, img):
        return np.squeeze(img)

    def cnn_filter(self, datas):
        """
        Run the normalised data through the noise cancellation CNN filter
        """
        # if use_normalization keep range for norm/renorm
        # model works with [0,1]
        try:
            # try a Keras model
            x = self.to_tensor(datas)
            y = self.model.predict(x)
            output = self.from_tensor(y)
        except AttributeError:
            # assume it is the cv.dnn_Net object
            x = cv.dnn.blobFromImage(datas.astype(np.float32), scalefactor=1.0,
                                     size=(datas.shape[1], datas.shape[0]), mean=0, swapRB=False)
            # model.setInput(x, name='input')
            self.model.setInput(x)
            # y = model.forward(outputName='subtract_1/sub')
            y = self.model.forward()
            output = np.squeeze(y, axis=(0, 1))
        return output

    def run_cnn(self, frames, min_temp, max_temp):
        '''plural names used for dinstingtion, are singular'''
        if self.use_normalization == 1:
            frames = remap(frames, curr_range=(min_temp, max_temp),
                           new_range=(0, 1), to_uint8=False)
            frames = self.cnn_filter(frames)
            frames = remap(frames, curr_range=(frames.min(), frames.max()),
                           new_range=(min_temp, max_temp), to_uint8=False)

        elif self.use_normalization == 2:
            frames = remap(frames, curr_range=(min_temp, max_temp),
                           new_range=(-1, 1), to_uint8=False)
            frames = self.cnn_filter(frames)
            frames = remap(frames, curr_range=(frames.min(), frames.max()),
                           new_range=(min_temp, max_temp), to_uint8=False)

        else:
            frames = self.cnn_filter(frames)
            frames = remap(frames, curr_range=(frames.min(), frames.max()),
                           new_range=(min_temp, max_temp), to_uint8=False)

        return frames

    def remap_tensor(self, data, new_range=(0, 1), axis_totake=(1, 2)):
        lo2, hi2 = new_range

        hi = np.max(data, axis=axis_totake, keepdims=True)[0]
        lo = np.min(data, axis=axis_totake, keepdims=True)[0]
        data -= lo
        data /= (hi - lo)
        data = lo2 + data * (hi2 - lo2)

        return data

    def predict(self):

        # Estimate frame update from previous frame and control params

        if self.which_predict == 'Median':
            self.est = remap(cv.medianBlur(remap(self.est, curr_range=(self.min_temp, self.max_temp)),
                                            ksize=self.ksize),
                             new_range=(self.min_temp, self.max_temp), to_uint8=False)
        elif self.which_predict == 'Gaussian':
            self.est = remap(cv.GaussianBlur(remap(self.est, curr_range=(self.min_temp, self.max_temp)),
                                              (self.ksize, self.ksize), sigmaX=self.sigma),
                             new_range=(self.min_temp, self.max_temp), to_uint8=False)
        elif self.which_predict == 'Box':
            self.est = remap(cv.blur(remap(self.est, curr_range=(self.min_temp, self.max_temp)),
                                      (self.ksize, self.ksize)),
                             new_range=(self.min_temp, self.max_temp), to_uint8=False)

        elif self.which_predict == 'CNN':
            self.est = self.run_cnn(self.est, self.min_temp, self.max_temp)

    def predict_new_frame(self, new_frame):
        # Estimate frame update from previous frame and control params
        max_temp = self.max_temp
        min_temp = self.min_temp

        if self.which_predict == 'Median':
            new_frame = remap(cv.medianBlur(remap(new_frame, curr_range=(min_temp, max_temp)),
                                             ksize=self.ksize))
            new_frame = remap(new_frame, curr_range=(new_frame.min(), new_frame.max()),
                           new_range=(min_temp, max_temp), to_uint8=False)
        elif self.which_predict == 'Gaussian':
            new_frame = remap(cv.GaussianBlur(remap(new_frame, curr_range=(min_temp, max_temp)),
                                              (self.ksize, self.ksize), sigmaX=self.sigma))
            new_frame = remap(new_frame, curr_range=(new_frame.min(), new_frame.max()),
                              new_range=(min_temp, max_temp), to_uint8=False)
        elif self.which_predict == 'Box':
            new_frame = remap(cv.blur(remap(new_frame, curr_range=(min_temp, max_temp)),
                                              (self.ksize, self.ksize)))
            new_frame = remap(new_frame, curr_range=(new_frame.min(), new_frame.max()),
                              new_range=(min_temp, max_temp), to_uint8=False)

        elif self.which_predict == 'CNN':
            new_frame = self.run_cnn(new_frame, min_temp, max_temp)

        return new_frame

    def update(self, new_frame):

        # # update roll average
        # self.min_temp += 1. / self.r_depth * (new_frame.min() - self.min_temp)
        # self.max_temp += 1. / self.r_depth * (new_frame.max() - self.max_temp)

        # no roll avg on new frame
        self.min_temp = new_frame.min()
        self.max_temp = new_frame.max()

        # predict current frame and process noise
        if self.count ==0:
            self.predict()
            self.count += 1
        if self.smooth_new_frame:
            new_frame_smooth = self.predict_new_frame(new_frame=new_frame)

        # calculate new actual error and new frame
        if self.update_which == 'l1':
            mea_err = np.abs(new_frame_smooth - self.est)
        elif self.update_which == 'l2':
            mea_err = (new_frame_smooth - self.est)**2

        if self.gain_which == 'sigmoid-approx':
            self.gain = self.gain_scale * (
                        1 + ((mea_err - self.pixel_err_std) / np.sqrt((1 + (mea_err - self.pixel_err_std) ** 2))))

        elif self.gain_which == 'normalizing':
            mea_err = mea_err + self.gain_scale
            self.gain = mea_err/(mea_err + self.pixel_err_std)

        elif self.gain_which == 'stair-case':
            # self.gain = 1/(1 + np.exp(-mea_err))
            self.gain = np.full(mea_err.shape, 0.8, dtype=float)
            self.gain[(mea_err <=2.0)] = 0.01
            self.gain[(mea_err > 2.0) & (mea_err <= 3.0)] = 0.1
            self.gain[(mea_err > 3.0) & (mea_err <= 5.0)] = 0.2
            self.gain[(mea_err > 5.0) & (mea_err <= 6.0)] = 0.3
            self.gain[(mea_err > 6.0) & (mea_err <= 10.)] = 0.5
            # self.gain[(mea_err > 3.0) & (mea_err <= 8.0)] = 0.7
            # # # self.gain[(mea_err > 5.0) & (mea_err <= 6.0)] = 0.5q
            # # # self.gain[(mea_err > 6.0) & (mea_err <=7.0)] = 0.6
            # # # self.gain[(mea_err > 7.0) & (mea_err <= 8.0)] = 0.7
            # # # self.gain[(mea_err > 8.0) & (mea_err <= 9.0)] = 0.8

        elif self.gain_which == 'poly':
            # self.gain = np.zeros_like(mea_err)
            self.gain = np.full((mea_err.shape), self.offset*self.scale)
            mea_err = (mea_err - self.pixel_err_std)*self.gain_scale
            self.gain[(mea_err > 0) & (mea_err <=1)] = (3*mea_err[(mea_err > 0) & (mea_err <=1)]**2
                                        - 2*mea_err[(mea_err > 0) & (mea_err <=1)]**3
                                        + self.offset)*self.scale
            self.gain[(mea_err>1)] = (1+self.offset)*self.scale

        # calculate new estimate
        if self.out_smooth:
            self.est = self.est + self.gain * (new_frame_smooth - self.est)
        else:
            self.est = self.est + self.gain * (new_frame - self.est)

        return self.est

    def __call__(self, new_frame, *args, **kwargs):
        return self.update(new_frame)
