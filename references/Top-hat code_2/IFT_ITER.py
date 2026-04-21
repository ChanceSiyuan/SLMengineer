# %%
from scipy.optimize import minimize
import numpy as np
import scipy as sp
import nlopt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import IMGpy
from skopt import gp_minimize,dump,load,forest_minimize
import slmpy
from scipy.special import laguerre,genlaguerre
from scipy.ndimage import gaussian_filter

from tqdm import tqdm 
import time
from scipy.ndimage import binary_dilation, rotate
import random
import torch
from skimage import measure, filters, morphology
import matplotlib.pyplot as plt
from skimage import filters, measure
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def calculate_centroid(image,threshold=80):
    """
    使用质心法计算图像中光斑的中心坐标
    :param image: 输入图像（需为灰度图）
    :return: (cx, cy) 质心坐标
    """
    height, width = image.shape
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    
    total_intensity = np.sum(image)
    # print("总光强：",total_intensity)        
    if total_intensity < threshold:
        raise ValueError("图像中未检测到光斑")
    cx = np.sum(xx * image) / total_intensity
    cy = np.sum(yy * image) / total_intensity
    return (cx, cy)

def d4sigma_centroid(image,threshold=0.05):
    image=np.array(image)
    h, w = image.shape
    x, y = np.arange(w), np.arange(h)

    image = image.astype(np.float64)
    image -= np.min(image)
    image[image < threshold * np.max(image)] = 0
    total_intensity = np.sum(image)

    (x0,y0)=calculate_centroid(image)
    sigma_x = np.sqrt(np.sum((x - x0)**2 * np.sum(image, axis=0)) / total_intensity)
    sigma_y = np.sqrt(np.sum((y - y0)**2 * np.sum(image, axis=1)) / total_intensity)
    x_min = max(0, int(x0 - 2 * sigma_x))
    x_max = min(w, int(x0 + 2 * sigma_x) + 1)
    y_min = max(0, int(y0 - 2 * sigma_y))
    y_max = min(h, int(y0 + 2 * sigma_y) + 1)
    cropped = image[y_min:y_max, x_min:x_max]

    (x0,y0)=calculate_centroid(cropped)
    x_center = x_min + x0
    y_center = y_min + y0

    return (x_center, y_center)   

def dsigma_centroid(image,threshold=0.05):
    h, w = image.shape
    x, y = np.arange(w), np.arange(h)

    image = image.astype(np.float64)
    image -= np.min(image)
    image[image < threshold * np.max(image)] = 0
    total_intensity = np.sum(image)

    (x0,y0)=calculate_centroid(image)
    sigma_x = np.sqrt(np.sum((x - x0)**2 * np.sum(image, axis=0)) / total_intensity)
    sigma_y = np.sqrt(np.sum((y - y0)**2 * np.sum(image, axis=1)) / total_intensity)
    x_min = max(0, int(x0 - 1 * sigma_x))
    x_max = min(w, int(x0 + 1 * sigma_x) + 1)
    y_min = max(0, int(y0 - 1 * sigma_y))
    y_max = min(h, int(y0 + 1 * sigma_y) + 1)
    cropped = image[y_min:y_max, x_min:x_max]

    (x0,y0)=calculate_centroid(cropped)
    x_center = x_min + x0
    y_center = y_min + y0

    return (x_center, y_center)   

def plot_3d(image):
    rows, cols = image.shape

    # 创建 X 和 Y 坐标
    x = np.arange(cols)  # 列索引为 X 坐标
    y = np.arange(rows)  # 行索引为 Y 坐标
    X, Y = np.meshgrid(x, y)

    # Z 值直接是你的矩阵
    Z = image

    # 创建 3D 图形
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制曲面图
    surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='k')

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # 设置标题和轴标签
    ax.set_title('3D Visualization of rotated_image')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # 显示图形
    plt.show()
def filtered_mean(data):
    """
    计算过滤掉均值±标准差范围外数据后的平均值
    参数：
        data : 输入数据列表（可包含int/float）
    返回：
        过滤后数据的平均值（若无有效数据返回None）
    """
    # 转换为NumPy数组提升计算效率
    arr = np.array(data, dtype=np.float64)
    
    # 计算均值和标准差（默认使用总体标准差ddof=0）
    mean = np.mean(arr)
    std = np.std(arr)
    
    # 确定过滤范围 [mean - std, mean + std]
    lower_bound = mean - std
    upper_bound = mean + std
    
    # 过滤数据（保留范围内的值）
    filtered = arr[(arr >= lower_bound) & (arr <= upper_bound)]
    
    # 处理无有效数据情况
    if len(filtered) == 0:
        print("警告：过滤后数据为空！")
        return np.mean(arr)
    
    # 返回过滤后数据的平均值
    return np.mean(filtered)


def hist_mean(data):
    data = np.array(data)  
    n_top = 3              

    # 计算直方图
    hist, bin_edges = np.histogram(data)

    top_bins_idx = np.argsort(hist)[-n_top:][::-1]

    top_bins_data = []
    for idx in top_bins_idx:
        bin_range = (bin_edges[idx], bin_edges[idx+1])
        bin_data = data[(data >= bin_range[0]) & (data < bin_range[1])]
        top_bins_data.append(bin_data.tolist())  # 转为list

    flat_list = [item for sublist in top_bins_data for item in sublist]
    return filtered_mean(flat_list)

def filtered_mean3(data):
    """
    计算过滤掉均值±标准差范围外数据后的平均值
    参数：
        data : 输入数据列表（可包含int/float）
    返回：
        过滤后数据的平均值（若无有效数据返回None）
    """
    # 转换为NumPy数组提升计算效率
    arr = np.array(data, dtype=np.float64)
    
    # 计算均值和标准差（默认使用总体标准差ddof=0）
    mean = np.mean(arr)
    std = np.std(arr)
    
    # 确定过滤范围 [mean - std, mean + std]
    lower_bound = mean - 3*std
    upper_bound = mean + 3*std
    
    # 过滤数据（保留范围内的值）
    filtered = arr[(arr >= lower_bound) & (arr <= upper_bound)]
    
    # 处理无有效数据情况
    if len(filtered) == 0:
        print("警告：过滤后数据为空！")
        return None
    
    # 返回过滤后数据的平均值
    return np.mean(filtered)

import sys
from PyQt5.QtWidgets import *
from CamOperation_class import CameraOperation
from MvCameraControl_class import *
from MvErrorDefine_const import *
from CameraParams_header import *
from PyUICBasicDemo import Ui_MainWindow
import ctypes
import matplotlib.pyplot as plt
# 将返回的错误码转换为十六进制显示
def ToHexStr(num):
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr
    return hexStr

# Decoding Characters
def decoding_char(c_ubyte_value):
    c_char_p_value = ctypes.cast(c_ubyte_value, ctypes.c_char_p)
    try:
        decode_str = c_char_p_value.value.decode('gbk')  # Chinese characters
    except UnicodeDecodeError:
        decode_str = str(c_char_p_value.value)
    return decode_str

cam = MvCamera()
global isOpen
isOpen = False

def enum_devices():
        global deviceList
        global obj_cam_operation

        deviceList = MV_CC_DEVICE_INFO_LIST()
        n_layer_type = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE
                        | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
        ret = MvCamera.MV_CC_EnumDevices(n_layer_type, deviceList)

        if ret != 0:
            strError = "Enum devices fail! ret = :" + ToHexStr(ret)
            return ret

        if deviceList.nDeviceNum == 0:
            return ret
        print("Find %d devices!" % deviceList.nDeviceNum)

        devList = []
        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE or mvcc_dev_info.nTLayerType == MV_GENTL_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                user_defined_name = decoding_char(mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName)
                model_name = decoding_char(mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName)
                print("device user define name: " + user_defined_name)
                print("device model name: " + model_name)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print("current ip: %d.%d.%d.%d " % (nip1, nip2, nip3, nip4))
                devList.append(
                    "[" + str(i) + "]GigE: " + user_defined_name + " " + model_name + "(" + str(nip1) + "." + str(
                        nip2) + "." + str(nip3) + "." + str(nip4) + ")")
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print("\nu3v device: [%d]" % i)
                user_defined_name = decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName)
                model_name = decoding_char(mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName)
                print("device user define name: " + user_defined_name)
                print("device model name: " + model_name)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print("user serial number: " + strSerialNumber)
                devList.append("[" + str(i) + "]USB: " + user_defined_name + " " + model_name
                               + "(" + str(strSerialNumber) + ")")
            elif mvcc_dev_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
                print("\nCML device: [%d]" % i)
                user_defined_name = decoding_char(mvcc_dev_info.SpecialInfo.stCMLInfo.chUserDefinedName)
                model_name = decoding_char(mvcc_dev_info.SpecialInfo.stCMLInfo.chModelName)
                print("device user define name: " + user_defined_name)
                print("device model name: " + model_name)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stCMLInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print("user serial number: " + strSerialNumber)
                devList.append("[" + str(i) + "]CML: " + user_defined_name + " " + model_name
                               + "(" + str(strSerialNumber) + ")")
            elif mvcc_dev_info.nTLayerType == MV_GENTL_CXP_DEVICE:
                print("\nCXP device: [%d]" % i)
                user_defined_name = decoding_char(mvcc_dev_info.SpecialInfo.stCXPInfo.chUserDefinedName)
                model_name = decoding_char(mvcc_dev_info.SpecialInfo.stCXPInfo.chModelName)
                print("device user define name: " + user_defined_name)
                print("device model name: " + model_name)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stCXPInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print("user serial number: " + strSerialNumber)
                devList.append("[" + str(i) + "]CXP: " + user_defined_name + " " + model_name
                               + "(" + str(strSerialNumber) + ")")
            elif mvcc_dev_info.nTLayerType == MV_GENTL_XOF_DEVICE:
                print("\nXoF device: [%d]" % i)
                user_defined_name = decoding_char(mvcc_dev_info.SpecialInfo.stXoFInfo.chUserDefinedName)
                model_name = decoding_char(mvcc_dev_info.SpecialInfo.stXoFInfo.chModelName)
                print("device user define name: " + user_defined_name)
                print("device model name: " + model_name)

                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stXoFInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print("user serial number: " + strSerialNumber)
                devList.append("[" + str(i) + "]XoF: " + user_defined_name + " " + model_name
                               + "(" + str(strSerialNumber) + ")")
                
def open_device(nSelCamIndex):
    global deviceList
    global obj_cam_operation
    global isOpen
    if isOpen:
        return MV_E_CALLORDER

    if nSelCamIndex < 0:
        return MV_E_CALLORDER

    obj_cam_operation = CameraOperation(cam, deviceList, nSelCamIndex)
    ret = obj_cam_operation.Open_device()
    if 0 != ret:
        strError = "Open device failed ret:" + ToHexStr(ret)
        print(strError)
        isOpen = False
        return ret
    else:
        set_continue_mode()

        get_param()

        isOpen = True

def close_device():
    global isOpen
    global isGrabbing
    global obj_cam_operation

    if isOpen:
        obj_cam_operation.Close_device()
        isOpen = False

    isGrabbing = False

def set_continue_mode():
    ret = obj_cam_operation.Set_trigger_mode(False)
    if ret != 0:
        strError = "Set continue mode failed ret:" + ToHexStr(ret)
        print(strError)
        return ret
    
def set_software_trigger_mode():
    ret = obj_cam_operation.Set_trigger_mode(True)
    if ret != 0:
        strError = "Set trigger mode failed ret:" + ToHexStr(ret)
        print(strError)
        return ret

def trigger_once():
    ret = obj_cam_operation.Trigger_once()
    if ret != 0:
        strError = "TriggerSoftware failed ret:" + ToHexStr(ret)
        print(strError)
        return ret
    
def get_param():
        ret = obj_cam_operation.Get_parameter()
        if ret != MV_OK:
            strError = "Get param failed ret:" + ToHexStr(ret)
            print(strError)
            return ret
        else:
            print("ExposureTime: {0:.2f}".format(obj_cam_operation.exposure_time))
            print("Gain: {0:.2f}".format(obj_cam_operation.gain))
            print("FrameRate: {0:.2f}".format(obj_cam_operation.frame_rate))
            print('PixelFormat: ',obj_cam_operation.pixel)

def set_param(frame_rate = None, exposure = None, gain = None):
    
    if frame_rate is None:
        frame_rate = obj_cam_operation.frame_rate
    if exposure is None:
        exposure = obj_cam_operation.exposure_time
    if gain is None:
        gain = obj_cam_operation.gain

    ret = obj_cam_operation.Set_parameter(frame_rate, exposure, gain)
    if ret != MV_OK:
        strError = "Set param failed ret:" + ToHexStr(ret)
        print(strError)
        return ret

    return MV_OK

def start_grabbing():
    global obj_cam_operation
    global isGrabbing

    ret = obj_cam_operation.start_grabbing()
    if ret != 0:
        strError = "Start grabbing failed ret:" + ToHexStr(ret)
        # print(strError)
        return ret
    else:
        isGrabbing = True

def stop_grabbing():
    global obj_cam_operation
    global isGrabbing
    ret = obj_cam_operation.stop_grabbing()
    if ret != 0:
        strError = "Stop grabbing failed ret:" + ToHexStr(ret)
        print(strError)
        return ret
    else:
        isGrabbing = False

def get_image():
    arr = obj_cam_operation.Get_img()
    
    # plt.imshow(arr)
    return arr

def center_max_region(image):
    thresh = filters.threshold_otsu(image)
    binary = image > thresh
    binary = morphology.remove_small_objects(binary, min_size=300)

    labels = measure.label(binary)
    props = measure.regionprops(labels)
    if not props:
        plt.imshow(image,cmap='jet')
        plt.show()
        center=d4sigma_centroid(image)
        return center[0],center[1]
        raise ValueError("未检测到主光斑")
    largest = max(props, key=lambda x: x.area)
    cy, cx = largest.centroid
    return cx,cy

def test_image():
    image=get_image()
    image=image[1480:1650,1880:2020]
    rotated_image = rotate(image,angle=11.3,reshape=False)
    # plt.imshow(rotated_image,cmap='jet')
    # plt.show()

    center=center_max_region(rotated_image)
    # center=dsigma_centroid(rotated_image)

    # rotated_image=rotated_image[int(center[1]-10):int(center[1]+11),int(center[0]-29):int(center[0]+30)]
    # cost=1e6*(1-abs(normalized_correlation(rotated_image,target)))

    rotated_image=rotated_image[int(center[1]-13):int(center[1]+14),int(center[0]-35):int(center[0]+36)]
    # cost=1e6*(1-abs(sliding_max_ncc(target,rotated_image)))
    plt.imshow(rotated_image,cmap='jet')
    plt.show()
    print(rotated_image.max())
    # cost=1e6*(1-abs(calculate_ssim(win_size=3,target=target,actual=rotated_image)))
    plot_3d(rotated_image)
#%%
enum_devices()
open_device(0)
# reset_roi()
start_grabbing()
#%%
slmx2 = slmpy.SLMdisplay(monitor=0,isImageLock = True)
#%%
# target=np.load('I_out_L_nocor_curv=1.2_guassian=30_Wcg=60.npy')
# path='L_nocor_curv=1.2_guassian=30_Wcg=60.npy'
# path='corr2.npy'

target=np.load('I_testx.npy')
path='phase_120wcg.npy'

# target=np.load('I_out_L_nocor_curv=1.2_d=80_guassian=30.npy')
# path='L_nocor_curv=1.2_d=80_guassian=30.npy'
path='phaseold_wcg=99.9_guassianx=29.600000381469727_guassiany=20.350000381469727.npy'
# path='phaseold_wcg=100_guassianx=30_guassiany=20_ir5.npy'
# path='phaseold_wcg=100_guassianx=30_guassiany=.npy'
# path='test_phase5.npy'
target=np.load('I_phaseold_wcg=99.9_guassianx=29.600000381469727_guassiany=20.350000381469727.npy')
# path='813-7168-0708-02.npy'
# path='813_eff5.npy'
tophatscreen=(255*np.load(path)/(2*np.pi))
# tophatscreen=(255*generate_tilted_grating([1024,1272],angle_deg=0)/(2*np.pi))

# tophatscreen=(255*phase/(2*np.pi))
# tophatscreen=255*np.array(cg1.slm_phase_end)/(2*np.pi)
tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
# tophat_screen_Corrected=(215*tophatscreen/255).astype('uint8')
slmx2.updateArray(tophat_screen_Corrected)
# %%
test_image()
# %%
def laser_gaussian(dim, r0, sigmax, sigmay, A=1.0, save_param=False):
    """
    Create n x n target:
    Gaussian laser beam profile centered on r0 = (x0,y0) with widths
    'sigmax' and 'sigmay' and amplitude 'A'
    """
    # initialization
    cols, rows = dim

    # Initialization
    x = torch.arange(rows, dtype=torch.float32) - rows / 2
    y = torch.arange(cols, dtype=torch.float32) - cols / 2
    X, Y = torch.meshgrid(x, y, indexing='xy')
    sigmax = torch.sqrt(torch.tensor(2.0)) * sigmax
    sigmay = torch.sqrt(torch.tensor(2.0)) * sigmay

    # target definition
    z = A*torch.exp( -2*(torch.pow((X-r0[0])/sigmax,2) + torch.pow((Y-r0[1])/sigmay,2) ) )

    if save_param :
        param_used = "target_gaussian | n={0} | r0={1} | sigmax={2} | sigmay={3} | A={4} ".format(rows, r0, sigmax, sigmay, A)
        return z, param_used
    else :
        return z

pixelpitch=12.5
SLMResX=1272
SLMResY=1024
beamwaist=3.5e3
w0=beamwaist/pixelpitch
L, Lp = laser_gaussian(dim=[1024,1272], r0=(0, 0), sigmax=beamwaist/pixelpitch, sigmay=beamwaist/pixelpitch, A=1.0, save_param=True)
# mask=create_circular_mask(1024,1272,radius=5e3/pixelpitch)
# L=L*mask
I_L_tot = torch.sum(torch.pow(L, 2.))                           #
L = L * torch.pow(10000 / I_L_tot, 0.5)                      #
I_L_tot = torch.sum(torch.pow(L, 2.))   

#%%
class InverseFourierOp:
    def __init__(self):
        pass

    def make_node(self, xr, xi):
        xr = torch.as_tensor(xr)
        xi = torch.as_tensor(xi)
        return xr, xi

    def perform(self, node, inputs, output_storage):
        xr, xi = inputs
        x = xr + 1j * xi
        nx, ny = xr.shape
        s = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x))) * (nx * ny)
        # 修改这里：直接赋值给output_storage
        output_storage[0] = s.real
        output_storage[1] = s.imag

    def __call__(self, xr, xi):
        # 创建节点
        inputs = self.make_node(xr, xi)
        # 修改这里：创建正确的output_storage结构
        output_storage = [None, None]  # 不需要预先创建tensor
        # 执行逆傅里叶变换
        self.perform(None, inputs, output_storage)
        return output_storage[0], output_storage[1]
class FourierOp:
    __props__ = ()
    
    def make_node(self, xr, xi):
        xr = torch.as_tensor(xr)
        xi = torch.as_tensor(xi)
        return xr, xi  # 返回输入张量

    def perform(self, node, inputs, output_storage):
        x = inputs[0] + 1j * inputs[1]
        s = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))
        output_storage[0] = s.real
        output_storage[1] = s.imag
        z_r = output_storage[0]
        z_i = output_storage[1]
        
    def grad(self, inputs, output_gradients):
        z_r = output_gradients[0]
        z_i = output_gradients[1]
        y = InverseFourierOp()(z_r, z_i)
        return y

    def __call__(self, xr, xi):
        # 创建节点
        inputs = self.make_node(xr, xi)
        output_storage = [torch.empty_like(xr), torch.empty_like(xi)]
        # 执行傅里叶变换
        self.perform(None, inputs, output_storage)
        return output_storage[0], output_storage[1]  # 返回实部和虚部
def get_centre_range(n,N):
    # returns the indices to use given an nxn SLM
    return int(N/2)-int(n/2),int(N/2)+int(n/2)
#%%
def dFFT(slm_phase):
    slm_phase=np.array(slm_phase)
    NT=[7030,7030]
    N=[1024,1272]
    profile_s=L
    if not torch.is_complex(profile_s):
        profile_s = profile_s.to(torch.complex128)
    n_pixelsx = int(N[0])
    n_pixelsy = int(N[1]) 
    profile_s_r = profile_s.real.type(torch.float64)
    profile_s_i = profile_s.imag.type(torch.float64)
    A0 = 1. / np.sqrt(NT[0] * NT[1])  # Linked to the fourier transform. Keeps the same quantity of light between the input and the output
    zero_frame = torch.zeros((NT[0], NT[1]), dtype=torch.float64)
    zero_matrix = torch.as_tensor(zero_frame, dtype=torch.float64)
    phi_reshaped = torch.from_numpy(slm_phase)
    S_r = profile_s_r.clone().detach().to(torch.float64)
    S_i = profile_s_i.clone().detach().to(torch.float64)
    E_in_r = A0 * (S_r * torch.cos(phi_reshaped) - S_i * torch.sin(phi_reshaped))
    E_in_i = A0 * (S_i * torch.cos(phi_reshaped) + S_r * torch.sin(phi_reshaped))
    idx_0x, idx_1x = get_centre_range(n_pixelsx,NT[0])
    idx_0y, idx_1y = get_centre_range(n_pixelsy,NT[1])
    E_in_r_pad = zero_matrix.clone()
    E_in_r_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_r  
    E_in_i_pad = zero_matrix.clone()
    E_in_i_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_i  
    phi_padded = zero_matrix.clone()
    phi_padded[idx_0x:idx_1x, idx_0y:idx_1y] = phi_reshaped

    E_in = E_in_r_pad + 1j * E_in_i_pad
    E_in_shifted = torch.fft.ifftshift(E_in)
    E_out = torch.fft.fft2(E_in_shifted)
    E_out_shifted = torch.fft.fftshift(E_out)
    E_out_r = E_out_shifted.real
    E_out_i = E_out_shifted.imag

    E_out_2 = E_out_r ** 2 + E_out_i ** 2  
    E_out_p = torch.atan2(E_out_i, E_out_r)  
    E_out_amp = torch.sqrt(E_out_2)
    return E_out_p,E_out_amp
def dIFFT(E_out_amp,E_out_p):
    NT=[7030,7030]
    N=[1024,1272]

    E_out_complex = E_out_amp * torch.exp(1j * E_out_p)
    
    E_out_shifted = torch.fft.ifftshift(E_out_complex)  
    E_in = torch.fft.ifft2(E_out_shifted)              
    E_in_shifted = torch.fft.fftshift(E_in)
    
    idx_0x, idx_1x = get_centre_range(N[0], NT[0])
    idx_0y, idx_1y = get_centre_range(N[1], NT[1])
    E_in_cropped = E_in_shifted[idx_0x:idx_1x, idx_0y:idx_1y]*torch.sqrt(torch.tensor(NT[0]*NT[1]))
    E_in_r = E_in_cropped.real
    E_in_i = E_in_cropped.imag
    
    E_in_2 = E_in_r ** 2 + E_in_i ** 2
    E_in_amp = torch.sqrt(E_in_2)
    E_in_p = torch.atan2(E_in_i, E_in_r)
    
    return E_in_p, E_in_amp
#%%
def pad_to_center(image_target0, target_shape, fill_value=0):
    """
    将 image_target0 填充到 target_shape，image_target0 居中
    支持 numpy 和 torch
    """
    h, w = image_target0.shape
    H, W = target_shape

    # 计算左上角起始点
    y0 = (H - h) // 2
    x0 = (W - w) // 2

    # 创建目标全零（或指定值）数组
    if isinstance(image_target0, torch.Tensor):
        padded = torch.full((H, W), fill_value, dtype=image_target0.dtype, device=image_target0.device)
    else:
        padded = np.full((H, W), fill_value, dtype=image_target0.dtype)

    # 填充
    padded[y0:y0+h, x0:x0+w] = image_target0
    return padded
def gs_vortex_correction(slm_phase,image_target, max_iter=100, tol=0.01, show=False):
    x,y=image_target.shape
    E_out_p,E_out_amp_init=dFFT(slm_phase)
    centerx=d4sigma_centroid(E_out_amp_init)
    imagex=E_out_amp_init.clone()
    E_out_amp_init_cut=E_out_amp_init[round(centerx[1]-x//2):round(centerx[1]+x//2+1),round(centerx[0]-y//2):round(centerx[0]+y//2+1)]
    image_target=torch.tensor(image_target.copy())*torch.sqrt((E_out_amp_init_cut**2).sum()/((image_target**2).sum()))
    imagex[round(centerx[1]-x//2):round(centerx[1]+x//2+1),round(centerx[0]-y//2):round(centerx[0]+y//2+1)]=image_target
    pre_error=np.inf

    for i in range(max_iter):
        E_in_p,E_in_amp=dIFFT(imagex,E_out_p)
        E_out_p,E_out_amp=dFFT(E_in_p)
        error = np.linalg.norm(E_out_amp - imagex) / np.linalg.norm(imagex)
        if show and i % 1 == 0:
            print(f'Iter {i}, error={error:.4f}')
        if np.abs( error-pre_error) < tol:
            print(f'Converged at iter {i}, error={error:.4f}')
            break
        pre_error=error

    return -(np.array(E_in_p)-slm_phase)
#%%
rep=300
slm_phase=np.load(path)
tophatscreen=(255*slm_phase/(2*np.pi))
tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
slmx2.updateArray(tophat_screen_Corrected)
time.sleep(0.5)
x=31
y=71
# x=21-12
# y=61-5
#%%
img_list=[]
for i in range(rep):
    images=get_image()
    # images=images[1650:1830,1780:1920]
    images=images[1480:1650,1880:2020]
    images = rotate(images,angle=11.3,reshape=False)
    image=np.flip(images, axis=0)
# plt.imshow(image)
# print(image.max())
    center=center_max_region(image)
    img_list.append(image.copy()[round(center[1])-x//2:round(center[1])+x//2+1,round(center[0])-y//2:round(center[0])+y//2+1])
    time.sleep(0.1)
# image_target=np.sqrt(image[int(center[1]-20):int(center[1]+21),int(center[0]-20):int(center[0]+21)])
avg_image = np.zeros_like(img_list[0], dtype=np.float32)
for img in img_list:
    avg_image += img.astype(np.float32) / rep
# avg_image=sigma_image(img_list)
image_target=np.sqrt(avg_image)

#%%
update_phase=slm_phase
correct_sum=[]
z_list=[]
best_cost=1
best_phase=0*slm_phase
#%%
for i in range(20):
    correct=gs_vortex_correction(slm_phase,image_target, max_iter=4, tol=0.0003, show=True)
    # 替换为新函数的调用
    # correct = phase_retrieval_optimizer(
    # slm_phase=slm_phase,
    # image_target=image_target,
    # max_iter=8,          # 增加总迭代次数，让HIO有足够时间精修
    # algorithm='GS_HIO',   # 使用推荐的混合模式
    # beta=0.9,             # 设置HIO反馈系数
    # switch_iter=2,       # 前20次用GS，后60次用HIO
    # use_filter=True,      # 对最终结果进行平滑
    # filter_sigma=1.5
    # )
    # correct=phase_retrieval_spot(image_target,lr=0.5, rep=200)
    # correct=gs_zeropad_correction(slm_phase,image_target, max_iter=5, tol=0.0006, show=True)

    # correct=HIO_correction(slm_phase,image_target, max_iter=60, tol=0.0006, show=True)
    # correct = gaussian_filter(correct, sigma=2)

    update_phase=update_phase+correct
    tophatscreen=(255*np.mod((update_phase),2*np.pi)/(2*np.pi))
    correct_sum.append(update_phase.copy())
    tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
    slmx2.updateArray(tophat_screen_Corrected)
    time.sleep(1.8)


    # images=get_image()
    # images=images[1725:1800,1730:1830]
    # images=images[1710:1800,1780:1850]
    # images = rotate(images,angle=34,reshape=False)
    # image=np.flip(images, axis=0)
    # plt.imshow(image)
    # print(image.max())
    # center=center_max_region(image)
    # plot_3d(image[int(center[1]-13):int(center[1]+14),int(center[0]-35):int(center[0]+36)])
    # image_target=np.sqrt(image[int(center[1]-13):int(center[1]+14),int(center[0]-35):int(center[0]+36)])
    img_list=[]
    test_image()
    for i in range(rep):
        images=get_image()
        images=images[1480:1650,1880:2020]
        # images=images[1650:1830,1780:1920]
        images = rotate(images,angle=11.3,reshape=False)
        image=np.flip(images, axis=0)
    # plt.imshow(image)
    # print(image.max())
        center=center_max_region(image)
        img_list.append(image.copy()[round(center[1])-x//2:round(center[1])+x//2+1,round(center[0])-y//2:round(center[0])+y//2+1])
        time.sleep(0.1)
    # image_target=np.sqrt(image[int(center[1]-20):int(center[1]+21),int(center[0]-20):int(center[0]+21)])
    avg_image = np.zeros_like(img_list[0], dtype=np.float32)
    for img in img_list:
        avg_image += img.astype(np.float32) / rep
    # avg_image=sigma_image(img_list)
    plot_3d(avg_image)
    # plt.imshow(avg_image,cmap='jet')
    # plt.show()
    z=avg_image[int(x//2),int(y//2-20):int(y//2+21)]
    print(100*z.std()/z.mean())
    print(100*(z.max()-z.min())/z.mean())
    z_list.append(z.std()/z.mean())
    # plt.plot(z)
    # plt.show()
    if z.std()/z.mean()-min(z_list)<=1e-6:
        best_phase=np.mod((update_phase),2*np.pi).copy()
        best_cost=z.std()/z.mean()
    image_target=np.sqrt(avg_image)
    # image_target=avg_image



#%%
tophatscreen=(255*best_phase/(2*np.pi))
tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
slmx2.updateArray(tophat_screen_Corrected)
time.sleep(0.5)
x=21-12
y=61-5
#%%
img_list=[]
for i in range(rep):
    images=get_image()
    # images=images[1650:1830,1780:1920]
    images=images[1480:1650,1880:2020]
    images = rotate(images,angle=11.3,reshape=False)
    image=np.flip(images, axis=0)
# plt.imshow(image)
# print(image.max())
    center=center_max_region(image)
    img_list.append(image.copy()[round(center[1])-x//2:round(center[1])+x//2+1,round(center[0])-y//2:round(center[0])+y//2+1])
    time.sleep(0.1)
# image_target=np.sqrt(image[int(center[1]-20):int(center[1]+21),int(center[0]-20):int(center[0]+21)])
avg_image = np.zeros_like(img_list[0], dtype=np.float32)
for img in img_list:
    avg_image += img.astype(np.float32) / rep
# avg_image=sigma_image(img_list)
image_target=np.sqrt(avg_image)

#%%
update_phase=best_phase.copy()

#%%
for i in range(5):
    correct=gs_vortex_correction(slm_phase,image_target, max_iter=4, tol=0.00003, show=True)
    # 替换为新函数的调用
    # correct = phase_retrieval_optimizer(
    # slm_phase=slm_phase,
    # image_target=image_target,
    # max_iter=8,          # 增加总迭代次数，让HIO有足够时间精修
    # algorithm='GS_HIO',   # 使用推荐的混合模式
    # beta=0.9,             # 设置HIO反馈系数
    # switch_iter=2,       # 前20次用GS，后60次用HIO
    # use_filter=True,      # 对最终结果进行平滑
    # filter_sigma=1.5
    # )
    # correct=phase_retrieval_spot(image_target,lr=0.5, rep=200)
    # correct=gs_zeropad_correction(slm_phase,image_target, max_iter=5, tol=0.0006, show=True)

    # correct=HIO_correction(slm_phase,image_target, max_iter=60, tol=0.0006, show=True)
    correct_sum.append(correct)
    # correct = gaussian_filter(correct, sigma=2)

    update_phase=update_phase+correct
    tophatscreen=(255*np.mod((update_phase),2*np.pi)/(2*np.pi))
    tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
    slmx2.updateArray(tophat_screen_Corrected)
    time.sleep(1.8)


    # images=get_image()
    # images=images[1725:1800,1730:1830]
    # images=images[1710:1800,1780:1850]
    # images = rotate(images,angle=34,reshape=False)
    # image=np.flip(images, axis=0)
    # plt.imshow(image)
    # print(image.max())
    # center=center_max_region(image)
    # plot_3d(image[int(center[1]-13):int(center[1]+14),int(center[0]-35):int(center[0]+36)])
    # image_target=np.sqrt(image[int(center[1]-13):int(center[1]+14),int(center[0]-35):int(center[0]+36)])
    img_list=[]
    for i in range(rep):
        images=get_image()
        images=images[1480:1650,1880:2020]
        # images=images[1650:1830,1780:1920]
        images = rotate(images,angle=11.3,reshape=False)
        image=np.flip(images, axis=0)
    # plt.imshow(image)
    # print(image.max())
        center=center_max_region(image)
        img_list.append(image.copy()[round(center[1])-x//2:round(center[1])+x//2+1,round(center[0])-y//2:round(center[0])+y//2+1])
        time.sleep(0.1)
    # image_target=np.sqrt(image[int(center[1]-20):int(center[1]+21),int(center[0]-20):int(center[0]+21)])
    avg_image = np.zeros_like(img_list[0], dtype=np.float32)
    for img in img_list:
        avg_image += img.astype(np.float32) / rep
    # avg_image=sigma_image(img_list)
    plot_3d(avg_image)
    # plt.imshow(avg_image,cmap='jet')
    # plt.show()
    z=avg_image[int(x//2),int(y//2-20):int(y//2+21)]
    print(100*z.std()/z.mean())
    print(100*(z.max()-z.min())/z.mean())
    z_list.append(z.std()/z.mean())
    # plt.plot(z)
    # plt.show()
    if z.std()/z.mean()-min(z_list)<=1e-6:
        best_phase=update_phase
        best_cost=z.std()/z.mean()
    image_target=np.sqrt(avg_image)
    # image_target=avg_image
#%%
slm_phase=np.load(path)
tophatscreen=(255*np.mod(slm_phase+correct_sum[0],2*np.pi)/(2*np.pi))
tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
slmx2.updateArray(tophat_screen_Corrected)
time.sleep(1)
test_image()
# %%

def sigma_image(img_list):
    # 假设 image_stack 是 (rep, height, width) 的三维数组
    image_stack = np.array(img_list)

    # Sigma裁剪的参数
    n_sigma = 2.5 # 设置为几倍sigma之外算作离群值

    # 1. 计算初始的平均值和标准差
    mean = np.mean(image_stack, axis=0)
    std = np.std(image_stack, axis=0)

    # 2. 确定每个像素的有效范围
    lower_bound = mean - n_sigma * std
    upper_bound = mean + n_sigma * std

    # 3. 创建一个布尔掩码，标记出所有在范围内的“好点”
    # 我们需要将边界扩展维度以便和 image_stack 广播比较
    good_mask = (image_stack >= lower_bound[np.newaxis, :, :]) & \
                (image_stack <= upper_bound[np.newaxis, :, :])

    # 4. 创建一个带掩码的数组，将“坏点”标记为无效
    masked_stack = np.ma.masked_array(image_stack, mask=~good_mask)

    # 5. 沿着帧轴计算掩码数组的平均值，无效点会被自动忽略
    sigma_clipped_image = np.ma.mean(masked_stack, axis=0).data

    return sigma_clipped_image
#%%
# %%
def create_1d_grating(k, shape):
    height, width = shape
    # 创建一个和SLM像素对应的x坐标网格
    x_coords = np.arange(width)
    # 计算每个x坐标的相位
    phase_1d = np.mod(k * x_coords, 2 * np.pi)
    # 将一维相位扩展为整个SLM的二维相位图
    phase_2d = np.tile(phase_1d, (height, 1))
    return phase_2d

def display_phase_on_slm(phase_map):
    """
    将一个相位图显示到SLM上。
    这个辅助函数封装了您之前的显示流程。
    """
    # 将相位从 [0, 2*pi] 转换为 [0, 255] 的灰度图
    screen_8bit = (255 * phase_map / (2 * np.pi))
    # 应用您自己的屏幕校正
    screen_corrected = IMGpy.SLM_screen_Correct(screen_8bit)
    # 更新SLM显示
    slmx2.updateArray(screen_corrected)
    # 等待SLM响应
    time.sleep(0.5)

def measure_spot_position():
    """
    从相机获取图像并测量光斑的精确x坐标。
    为了结果稳定，这里进行一个简单的5帧平均。
    """
    images = []
    for _ in range(5):
        images.append(get_image())
        time.sleep(0.05)
    
    avg_image = np.mean(np.array(images), axis=0)
    center_x, center_y = d4sigma_centroid(avg_image)
    
    # 可选：显示图像以供调试
    # plt.imshow(avg_image, cmap='jet')
    # plt.scatter(center_x, center_y, c='red', marker='+')
    # plt.title(f"Measured Center: ({center_x:.2f}, {center_y:.2f})")
    # plt.show()
    
    return center_x

def run_calibration():

    SLM_SHAPE = (1024, 1272) # SLM的分辨率 (height, width)
    M_Holo = 1024            # 您在SLM上显示全息图的直径 (单位: 像素)
    M_Array = 1024           # 您进行FFT计算时数组的边长 (单位: 像素)

    # 定义两个光栅的空间频率 k1 和 k2 (单位: 弧度/像素)
    # 这个值的选择需要让光点在相机视野内有明显但不过大的移动
    k1 = 0.05
    k2 = 0.06
    delta_k = k2 - k1

    grating1 = create_1d_grating(k1, SLM_SHAPE)
    display_phase_on_slm(grating1)
    
    print("测量第一个光点位置 l1...")
    l1_x = measure_spot_position()
    print(f"  > 位置 l1_x = {l1_x:.2f} pixels")
    
    # --- 3. 测量第二个光点位置 ---
    print("\n显示第二个光栅 (k2)...")
    grating2 = create_1d_grating(k2, SLM_SHAPE)
    display_phase_on_slm(grating2)
    
    print("测量第二个光点位置 l2...")
    l2_x = measure_spot_position()
    print(f"  > 位置 l2_x = {l2_x:.2f} pixels")
    
    # --- 4. 计算结果 ---
    delta_l = abs(l2_x - l1_x)
    print(f"\n测量得到的光点位移: Δl = {delta_l:.2f} pixels")

    f=delta_l*1.85*1e-6/813*1e9/delta_k*2*np.pi*12.5*1e-6
    if delta_l < 1:
        print("\n!! 警告: 光点位移过小，可能导致结果不准确。请尝试增在 k1 和 k2 之间的差异。")
        return None

    # --- 5. 应用论文公式 (Eq. 4) 计算孔径 A ---
    A = (M_Holo * M_Array * delta_k) / (2 * np.pi * delta_l)
    
    print("\n--- 标定完成 ---")
    print(f"根据论文公式 (Eq. 4) 计算得出：")
    print(f"模拟所需孔径直径 A = {A:.2f} 像素")
    
    return A

# --- 如何运行 ---
# 在您的主代码中，完成相机和SLM的初始化后，调用此函数
# if __name__ == '__main__':
#     # ...您的相机和SLM初始化代码...
#
#     calibrated_aperture_A = run_calibration()
#
#     if calibrated_aperture_A is not None:
#         print(f"\n请在后续的优化算法中，使用直径为 {calibrated_aperture_A:.2f} 像素的圆形孔径。")
#         # 接下来就可以开始您的主优化循环了...
f=0.19590104157239902