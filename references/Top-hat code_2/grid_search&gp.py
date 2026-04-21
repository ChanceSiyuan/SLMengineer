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

def find_center_by_crosshair(image, show=False):
    """
    使用crosshair（投影找峰值）方法确定图像中心。
    对于强度不均但有对称轴的光斑很有效。

    :param image: 输入的二维图像 (numpy array)
    :param show: 是否显示投影和峰值结果
    :return: (cx, cy) 确定的中心坐标
    """
    # 1. 投影到X轴和Y轴
    proj_y = np.sum(image, axis=0)  # 垂直投影到X轴
    proj_x = np.sum(image, axis=1)  # 水平投影到Y轴

    # 2. 在投影上找峰值
    # 使用find_peaks可以找到一个或多个峰值，我们取强度最大的那个
    peaks_x_indices, properties_x = find_peaks(proj_x, height=0)
    if len(peaks_x_indices) == 0:
        cy = image.shape[0] / 2 # 如果没找到峰值，返回中心
    else:
        # 找到最高的峰
        cy = peaks_x_indices[np.argmax(properties_x['peak_heights'])]

    peaks_y_indices, properties_y = find_peaks(proj_y, height=0)
    if len(peaks_y_indices) == 0:
        cx = image.shape[1] / 2 # 如果没找到峰值，返回中心
    else:
        # 找到最高的峰
        cx = peaks_y_indices[np.argmax(properties_y['peak_heights'])]
    
    # 可视化（可选）
    if show:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        
        # 显示原图和中心
        axs[0, 0].imshow(image, cmap='hot')
        axs[0, 0].axvline(cx, color='cyan', linestyle='--')
        axs[0, 0].axhline(cy, color='cyan', linestyle='--')
        axs[0, 0].set_title("Image with Crosshair Center")

        # 显示Y轴投影和峰值
        axs[1, 0].plot(proj_y)
        axs[1, 0].axvline(cx, color='red', linestyle='--')
        axs[1, 0].set_title("Projection onto X-axis")

        # 显示X轴投影和峰值
        axs[0, 1].plot(proj_x, np.arange(len(proj_x)))
        axs[0, 1].axhline(cy, color='red', linestyle='--')
        axs[0, 1].invert_yaxis() # 翻转Y轴以匹配图像坐标
        axs[0, 1].set_title("Projection onto Y-axis")

        axs[1, 1].axis('off') # 隐藏右下角的空图
        
        plt.tight_layout()
        plt.show()

    return cx, cy

def create_circular_mask(h, w, center=None, radius=None):
    """
    生成一个圆形的掩码
    :param h: 图像高度
    :param w: 图像宽度
    :param center: (x0, y0)，圆心坐标，默认为图像中心
    :param radius: 半径（像素），默认为图像短边的一半
    :return: 圆形掩码（布尔型二维数组）
    """
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask
def fit_vortex_radius(image, center=None, threshold_ratio=0.3, show=True):
    """
    对涡旋光（圆环）图像拟合其半径
    :param image: 输入二维灰度图
    :param center: (x0, y0)，圆心坐标，默认为图像中心
    :param threshold_ratio: 阈值比例，自动分割圆环
    :param show: 是否显示拟合结果
    :return: 拟合半径
    """
    h, w = image.shape
    if center is None:
        x0, y0 = w // 2, h // 2
    else:
        x0, y0 = center

    # 归一化
    img = image.astype(np.float64)
    img -= img.min()
    img /= (img.max() + 1e-8)

    # 阈值分割，提取圆环
    thresh = threshold_ratio * img.mean()
    mask = img > thresh

    # 计算所有圆环像素到中心的距离
    y_idx, x_idx = np.nonzero(mask)
    r = np.sqrt((x_idx - x0) ** 2 + (y_idx - y0) ** 2)

    # 用中值或均值作为拟合半径
    radius = np.median(r)

    if show:
        plt.imshow(img, cmap='gray')
        circle = plt.Circle((x0, y0), radius, color='r', fill=False, linewidth=2)
        plt.gca().add_patch(circle)
        plt.title(f'拟合半径: {radius:.1f}')
        plt.show()

    return radius
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

def load_zernike_phase_on_slm(best_params, zernike_phases):
    """
    根据best_params在SLM加载相应的Zernike相位
    参数:
        best_params: list/ndarray, 最优参数
        zernike_phases: dict, 预计算的zernike相位
        base_phase: ndarray, 基础相位
    """
    total_phase = np.load(path).copy()
    for idx, value in enumerate(best_params):
        if value != 0:
            # 找到最接近的预计算值
            closest_value = min(zernike_phases[idx].keys(), key=lambda x: abs(x-value))
            total_phase += zernike_phases[idx][closest_value]
    total_phase = np.mod(total_phase, 2 * np.pi)
    tophatscreen = (255 * total_phase / (2 * np.pi))
    tophat_screen_Corrected = IMGpy.SLM_screen_Correct(tophatscreen)
    slmx2.updateArray(tophat_screen_Corrected)

def load_peak_on_slm(best_params):
    x1=best_params[0]
    x2=best_params[1]
    x3=best_params[2]
    x4=best_params[3]
    x5=best_params[4]
    x6=best_params[5]
    x7=best_params[6]
    x8=best_params[7]
    # if 1-x1**2-x2**2-x3**2-x4**2<0:
    #     return 1e10
    # x0=np.sqrt(1-x1**2-x2**2-x3**2-x4**2)
    phase2 = np.angle(
        np.exp(1j * phase_list[0]) +
        x1 * np.exp(1j * phase_list[1]) +
        x2 * np.exp(1j * phase_list[2]) +
        x3 * np.exp(1j * phase_list[3]) +
        x4 * np.exp(1j * phase_list[4])+
        x5 * np.exp(1j * phase_list[5]) +
        x6 * np.exp(1j * phase_list[6]) +
        x7 * np.exp(1j * phase_list[7]) +
        x8 * np.exp(1j * phase_list[8])
    )
    total_phase = np.mod(phase2, 2 * np.pi)
    tophatscreen = (255 * total_phase / (2 * np.pi))
    tophat_screen_Corrected = IMGpy.SLM_screen_Correct(tophatscreen)
    slmx2.updateArray(tophat_screen_Corrected)
from scipy.signal import correlate2d

def normalized_max_cross_correlation(img1, img2):
    """
    计算归一化最大互相关值（范围[-1,1]，1表示完全匹配）
    """
    # 去均值处理
    img1 = img1 - np.mean(img1)
    img2 = img2 - np.mean(img2)
    
    # 计算互相关分子
    numerator = correlate2d(img1, img2, mode='same')
    
    # 计算归一化分母
    denominator = np.sqrt(np.sum(img1**2) * np.sum(img2**2))
    
    # 避免除以零
    if denominator == 0:
        return 0.0
    
    # 计算归一化互相关矩阵
    ncc_matrix = numerator / denominator
    
    return np.max(ncc_matrix)

def calculate_mse(target, actual,a=2):
    if target.shape != actual.shape:
        print(f"尺寸不匹配！目标尺寸：{target.shape}，实际尺寸：{actual.shape}")
        return 200.0
    actual=actual*(target.sum())/(actual.sum())
    return np.sum((abs(target - actual))**a)

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

def calculate_psnr(target, actual):
    actual=actual*(target.sum())/(actual.sum())
    mse = np.mean((target - actual) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(target)
    return 20 * np.log10(max_pixel / np.sqrt(mse))
from skimage.metrics import structural_similarity as ssim
target_ideal = np.load('ideal-813-15-30-target.npy')
def calculate_ssim(win_size,target, actual):
    actual=actual*(target.sum())/(actual.sum())
    # 建议使用多通道版本处理复杂光场
    return ssim(target, actual, 
               win_size=win_size,  # 根据图像尺寸调整
               data_range=np.max(target)-np.min(target),
               gaussian_weights=True)

from skimage.util import view_as_windows
def speckle_contrast(image, window_size=7):
    # 滑动窗口计算局部对比度
    windows = view_as_windows(image, (window_size, window_size))
    means = np.mean(windows, axis=(2,3))
    stds = np.std(windows, axis=(2,3))
    return np.mean(stds / (means + 1e-6))  # 避免除以零

def contrast_difference(target, actual):
    actual=actual*(target.sum())/(actual.sum())

    return abs(speckle_contrast(target) - speckle_contrast(actual))
def add_zernike_phase(x0,indices):
    Zernike_phase = IMGpy.Zernike_Generate(SLMRes[0], SLMRes[1], pixelpitch, aperture_radius, indices, x0, isZernikePhaseContinous)
    phase = np.mod(np.load(path) + Zernike_phase, 2 * np.pi)
    tophatscreen = (255 * phase / (2 * np.pi))
    tophat_screen_Corrected = IMGpy.SLM_screen_Correct(tophatscreen)
    slmx2.updateArray(tophat_screen_Corrected)

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

# from skimage import exposure


from skimage.feature import match_template
#滑窗NCC
def sliding_max_ncc(target, actual):
    result = match_template(actual, target, pad_input=True)
    return np.max(result)


def normalized_correlation(img1, img2):
    # 归一化到0-1

    img1_norm = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
    img2_norm = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
    
    # 计算均值和标准差
    mu1, mu2 = np.mean(img1_norm), np.mean(img2_norm)
    sigma1, sigma2 = np.std(img1_norm), np.std(img2_norm)
    
    # 计算NCC
    numerator = np.sum((img1_norm - mu1) * (img2_norm - mu2))
    denominator = np.sqrt(np.sum((img1_norm - mu1)**2) * np.sum((img2_norm - mu2)**2))
    return numerator / denominator

# ncc_score = normalized_correlation(target, image)
# print(f"NCC Score: {ncc_score}")  # 范围 [-1, 1]，1表示完全一致
import pickle
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


# with open('zernike_phase_5.3.pkl', 'rb') as f:
#     zernike_phase = pickle.load(f)

# with open('zernike_phase_(-0.1,0.1)_0.001.pkl', 'rb') as f:
#     zernike_phase2 = pickle.load(f)
from scipy.stats import entropy
def calculate_mi(img1, img2, bins=100):
    """计算两幅图像的互信息"""
    # 归一化图像到[0,1]
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    
    # 创建联合直方图
    hist_2d, _, _ = np.histogram2d(
        img1.ravel(), 
        img2.ravel(), 
        bins=bins,
        range=[[0, 1], [0, 1]]
    )
    
    # 计算概率分布
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    # 计算互信息
    hx = entropy(px)
    hy = entropy(py)
    hxy = entropy(pxy.flatten())
    
    return hx + hy - hxy
def test_image():
    image=get_image()
    image=image[1000:1830,1780:1920]
    rotated_image = rotate(image,angle=12,reshape=False)
    # plt.imshow(rotated_image,cmap='jet')
    # plt.show()

    center=center_max_region(rotated_image)
    # center=dsigma_centroid(rotated_image)

    # rotated_image=rotated_image[int(center[1]-10):int(center[1]+11),int(center[0]-29):int(center[0]+30)]
    # cost=1e6*(1-abs(normalized_correlation(rotated_image,target)))

    rotated_image=rotated_image[int(center[1]-13):int(center[1]+14),int(center[0]-35):int(center[0]+36)]
    # cost=1e6*(1-abs(sliding_max_ncc(target,rotated_image)))
    cost=1e6*(1-abs(sliding_max_ncc(target[:,8:-8],rotated_image[:,8:-8])))
    plt.imshow(rotated_image,cmap='jet')
    plt.show()
    print(rotated_image.max())
    # cost=1e6*(1-abs(calculate_ssim(win_size=3,target=target,actual=rotated_image)))
    print(cost)
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
#%%
phase2=np.angle(np.exp(1j*(np.load(path)))+0.*np.exp(1j*(np.load('peak0.npy')))+0.*np.exp(1j*(np.load('peak3.npy')))+0.*np.exp(1j*(np.load('peak2.npy')))+0.*np.exp(1j*(np.load('peak1.npy'))))
tophatscreen=(255*phase2/(2*np.pi))
# tophatscreen=(255*generate_tilted_grating([1024,1272],angle_deg=0)/(2*np.pi))

# tophatscreen=(255*phase/(2*np.pi))
# tophatscreen=255*np.array(cg1.slm_phase_end)/(2*np.pi)
tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
# tophat_screen_Corrected=(215*tophatscreen/255).astype('uint8')
slmx2.updateArray(tophat_screen_Corrected)
# %%
test_image()
# %%
# phase_list = [
#     np.load(path),      # 请将'path.npy'替换为你的主相位文件名
#     np.load('peak0.npy'),
#     np.load('peak1.npy'),
#     np.load('peak2.npy'),
#     np.load('peak3.npy'),
#     np.load('peak4.npy'),
#     np.load('peak5.npy'),
#     np.load('peak6.npy'),
#     np.load('peak7.npy')
# ]
n = 12

phase_list = [np.load(path)] + [
    np.load(f'peak_12{i}.npy') for i in range(n)
]
pixelpitch = 12.5  # SLM像素尺寸，单位um
SLMRes = [1272, 1024]  # SLM屏幕分辨率
ind_Zernike_phase1 = [3, 4, 5, 6, 7, 8, 9]  # 第一阶段优化的Zernike项
ind_Zernike_phase2 = [12, 13, 14, 15, 16, 17, 18, 19, 20]  # 第二阶段优化的Zernike项
aperture_radius = 5.3e3  # 单位um
isZernikePhaseContinous = False
def rosenbrock_general(x):
    """通用的目标函数，根据传入的indices生成Zernike相位并计算损失"""
    # 使用和网格搜索一样的方式加载相位图
    total_phase = np.load(path).copy()
    for idx, value in enumerate(x):
        if value != 0:  # 跳过零值参数
            # 找到最接近的预计算值
            closest_value = min(zernike_phase[idx].keys(), 
                             key=lambda x_val: abs(x_val-value))
            total_phase += zernike_phase[idx][closest_value]
    
    total_phase = np.mod(total_phase, 2 * np.pi)
    tophatscreen = (255 * total_phase / (2 * np.pi))
    tophat_screen_Corrected = IMGpy.SLM_screen_Correct(tophatscreen)
    slmx2.updateArray(tophat_screen_Corrected)
    time.sleep(0.5)
    cost_samples=[]
    for v in range(30):
        image=get_image()
        image=image[1725:1800,1730:1830]
        rotated_image = rotate(image,angle=34,reshape=False)
        center=center_max_region(rotated_image)
        rotated_image=rotated_image[int(center[1]-13):int(center[1]+14),int(center[0]-35):int(center[0]+36)]
        if all(r >= t for r, t in zip(rotated_image.shape, target.shape)):
            cost=1e6*(1-abs(sliding_max_ncc(target,rotated_image)))
            cost_samples.append(cost)
        time.sleep(0.01)
    costx=hist_mean(cost_samples)
    if np.isnan(hist_mean(cost_samples)):
        costx=1e10
    return costx

def rosenbrock_general2(x):
    r,theta1,theta2,theta3=x[0],x[1],x[2],x[3]
    x1 = r * np.sin(theta1) * np.sin(theta2) * np.sin(theta3)
    x2 = r * np.sin(theta1) * np.sin(theta2) * np.cos(theta3)
    x3 = r * np.sin(theta1) * np.cos(theta2)
    x4 = r * np.cos(theta1)
    x0=np.sqrt(1-x1**2-x2**2-x3**2-x4**2)
    phase2 = np.angle(
        x0 * np.exp(1j * phase_list[0]) +
        x1 * np.exp(1j * phase_list[1]) +
        x2 * np.exp(1j * phase_list[2]) +
        x3 * np.exp(1j * phase_list[3]) +
        x4 * np.exp(1j * phase_list[4])
    )
    total_phase = np.mod(phase2, 2 * np.pi)
    tophatscreen = (255 * total_phase / (2 * np.pi))
    tophat_screen_Corrected = IMGpy.SLM_screen_Correct(tophatscreen)
    slmx2.updateArray(tophat_screen_Corrected)
    time.sleep(0.5)
    cost_samples=[]
    for v in range(60):
        image=get_image()
        image=image[1725:1800,1730:1830]
        rotated_image = rotate(image,angle=34,reshape=False)
        center=center_max_region(rotated_image)
        rotated_image=rotated_image[int(center[1]-13):int(center[1]+14),int(center[0]-35):int(center[0]+36)]
        if all(r >= t for r, t in zip(rotated_image.shape, target.shape)):
            cost=1e6*(1-abs(sliding_max_ncc(target,rotated_image)))
            cost_samples.append(cost)
        time.sleep(0.01)
    costx=hist_mean(cost_samples)
    if np.isnan(hist_mean(cost_samples)):
        costx=1e10
    return costx
def rosenbrock_general3(x):
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    x7=x[6]
    x8=x[7]
    # if 1-x1**2-x2**2-x3**2-x4**2<0:
    #     return 1e10
    # x0=np.sqrt(1-x1**2-x2**2-x3**2-x4**2)
    phase2 = np.angle(
        np.exp(1j * phase_list[0]) +
        x1 * np.exp(1j * phase_list[1]) +
        x2 * np.exp(1j * phase_list[2]) +
        x3 * np.exp(1j * phase_list[3]) +
        x4 * np.exp(1j * phase_list[4])+
        x5 * np.exp(1j * phase_list[5]) +
        x6 * np.exp(1j * phase_list[6]) +
        x7 * np.exp(1j * phase_list[7]) +
        x8 * np.exp(1j * phase_list[8])
    )
    total_phase = np.mod(phase2, 2 * np.pi)
    tophatscreen = (255 * total_phase / (2 * np.pi))
    tophat_screen_Corrected = IMGpy.SLM_screen_Correct(tophatscreen)
    slmx2.updateArray(tophat_screen_Corrected)
    time.sleep(0.5)
    cost_samples=[]
    for v in range(60):
        image=get_image()
        image=image[1725:1800,1730:1830]
        rotated_image = rotate(image,angle=34,reshape=False)
        center=center_max_region(rotated_image)
        rotated_image=rotated_image[int(center[1]-13):int(center[1]+14),int(center[0]-35):int(center[0]+36)]
        if all(r >= t for r, t in zip(rotated_image.shape, target.shape)):
            cost=1e6*(1-abs(sliding_max_ncc(target,rotated_image)))
            cost_samples.append(cost)
        time.sleep(0.01)
    costx=hist_mean(cost_samples)
    if np.isnan(hist_mean(cost_samples)):
        costx=1e10
    return costx
#%%
dimensions = [(-1.,1.)]*18
ind_Zernike_phase1 = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]  # 第一阶段优化的Zernike项
dimensions=[(0,1),(0,np.pi/2),(0,np.pi/2),(0,np.pi/2)]
#%%
def block_bayes_optimize(func, dimensions, block_indices, n_calls=100, n_initial_points=10):
    x_best = np.zeros(len(dimensions))
    for iter in range(3):  # 迭代轮数
        print(f"==== Block Iteration {iter+1} ====")
        for block in block_indices:
            # 只优化当前block，其余参数固定
            def partial_func(x_block):
                x = x_best.copy()
                for i, idx in enumerate(block):
                    x[idx] = x_block[i]
                return func(x)
            block_dims = [dimensions[i] for i in block]
            res = gp_minimize(
                partial_func,
                block_dims,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                random_state=42,
                acq_func="EI"
                # verbose=True
            )
            # 更新最优参数
            for i, idx in enumerate(block):
                x_best[idx] = res.x[i]
            print(f"Block {block} updated: {x_best}cost:{res.fun}")
    return x_best

import random
def random_block_bayes_optimize(func, dimensions, block_size=9, n_blocks=2, n_calls=100, n_initial_points=10, n_iter=5):
    n_vars = len(dimensions)
    x_best = np.zeros(n_vars)
    for iter in range(n_iter):  # 迭代轮数
        print(f"==== Random Block Iteration {iter+1} ====")
        # 本轮生成n_blocks个不重复的随机分块
        all_indices = list(range(n_vars))
        blocks = []
        used = set()
        for _ in range(n_blocks):
            # 保证每个block不完全重复（可选）
            while True:
                block = tuple(sorted(random.sample(all_indices, block_size)))
                if block not in used:
                    used.add(block)
                    blocks.append(list(block))
                    break
        for block in blocks:
            def partial_func(x_block):
                x = x_best.copy()
                for i, idx in enumerate(block):
                    x[idx] = x_block[i]
                return func(x)
            block_dims = [dimensions[i] for i in block]
            res = gp_minimize(
                partial_func,
                block_dims,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                random_state=42,
                acq_func="EI",
                verbose=True
            )
            for i, idx in enumerate(block):
                x_best[idx] = res.x[i]
            print(f"Block {block} updated: {x_best}")
    return x_best
#%%
block_indices = [
    list(range(0, 3)),
    list(range(3, 7)),   # 第一块：0~8
    list(range(7, 12)),
    list(range(12, 18))   # 第二块：9~17
]
best_params = block_bayes_optimize(rosenbrock_general, dimensions, block_indices)
print("最终最优参数：", best_params)
load_zernike_phase_on_slm(best_params, zernike_phase)
#%%


# result = gp_minimize(
#     func=rosenbrock_general,
#     dimensions=dimensions,
#     n_calls=300,
#     n_initial_points=40,
#     n_jobs=-1,
#     # x0=x,
#     # y0=y0,
#     # noise=0.1,  # 指定噪声方差（高斯过程自动建模）
#     random_state=42,
#     acq_func="EI",  # 使用期望改进采集函数 PI,LCB
#     verbose=True
# )

# result = forest_minimize(
#     func=rosenbrock_general,      # 你的目标函数
#     dimensions=dimensions,        # 参数空间
#     n_calls=200,                  # 总采样次数
#     n_initial_points=40,          # 初始随机采样数
#     base_estimator='RF',          # 'RF'为随机森林，'ET'为极端随机树
#     acq_func='EI',                # 采集函数：EI, PI, LCB
#     random_state=42,              # 随机种子
#     xi=0.01,                      # 控制探索程度（EI/PI时）
#     kappa=1.96,                    # 控制探索程度（LCB时）
#     verbose=True
# )
print("最优参数:", result.x)
print("最优值:", result.fun)

# X = np.array(result.x_iters)  # shape: (n_samples, n_params)
# y = np.array(result.func_vals)

# for i in range(X.shape[1]):
#     corr = np.corrcoef(X[:, i], y)[0, 1]
#     print(f"参数{i} 与目标值的相关系数: {corr:.3f}")

# from sklearn.decomposition import PCA

# X = np.array(result.x_iters)
# pca = PCA()
# pca.fit(X)
# print("每个主成分解释的方差比例：", pca.explained_variance_ratio_)
#%%
from skopt.plots import plot_convergence,plot_evaluations,plot_objective
import matplotlib.pyplot as plt

plot_convergence(result)
plt.show()
#%%
plot_evaluations(result)
plt.show()
plot_objective(result)
#%%
best_params=result.x
load_zernike_phase_on_slm(best_params, zernike_phase)
#%%
from skopt import dump

dump(result, 'gp_result.pkl', store_objective=True)

#%%
from skopt import load

result = load('gp_result.pkl')

#%%
result = gp_minimize(
    func=rosenbrock_general,
    dimensions=dimensions,
    n_calls=200,  # 新的总迭代次数（包含已有的）
    x0=result.x_iters,  # 传入上次的采样点
    y0=result.func_vals,  # 传入上次的目标值
    random_state=42,
    acq_func="EI",
    verbose=True
)
#%%


# %%
def precalculate_zernike_phases(param_ranges, step_sizes, indices):
    """
    预计算所有可能的Zernike相位图
    
    参数:
    param_ranges : list of tuples - 每个参数的搜索范围 (min, max)
    step_sizes : list of floats - 每个参数的步长
    indices : list of int - Zernike项的索引
    
    返回:
    dict : 以参数值为键,相应的相位图为值的字典
    """
    zernike_phases = {}
    
    # 为每个参数生成网格点
    param_grids = []
    for (param_min, param_max), step in zip(param_ranges, step_sizes):
        grid = np.arange(param_min, param_max + step/2, step)
        param_grids.append(grid)
    
    # 为每个参数单独计算相位图
    for param_idx, param_grid in enumerate(param_grids):
        param_phases = {}
        for value in param_grid:
            # 创建参数向量(其他参数为0)
            params = np.zeros(len(indices))
            params[param_idx] = value
            
            # 计算相位图
            phase = IMGpy.Zernike_Generate(
                SLMRes[0], SLMRes[1], 
                pixelpitch, 
                aperture_radius, 
                [indices[param_idx]], 
                [value], 
                isZernikePhaseContinous
            )
            param_phases[value] = phase
            
        zernike_phases[param_idx] = param_phases
    
    return zernike_phases

#%%
def grid_search_zernike_optimized(
    param_ranges,
    zernike_phase, 
    step_sizes, 
    max_iter=3,
    n_samples=5,
    early_stop_threshold=0.01):
    """
    优化后的网格搜索算法,使用预计算的Zernike相位图
    """
    # 预计算所有可能的Zernike相位图
    # zernike_phases = precalculate_zernike_phases(param_ranges, step_sizes, ind_Zernike_phase1)    
    zernike_phases = zernike_phase
    n_params = len(param_ranges)
    best_params = np.zeros(n_params)
    best_cost = float('inf')
    base_phase = np.load(path)
    phase_list = [np.load(path)] + [
    np.load(f'peak{i}.npy') for i in range(n_params)
]
    for iter in range(max_iter):
        print(f"\n=== Iteration {iter+1}/{max_iter} ===")
        cost_improved = False
        
        for param_idx in range(n_params):
            param_min, param_max = param_ranges[param_idx]
            step = step_sizes[param_idx]
            candidates = np.arange(param_min, param_max+step/2, step)
            
            current_best_val = best_params[param_idx]
            current_best_cost = best_cost
            
            for candidate in tqdm(candidates, desc=f"Param {param_idx}"):
                test_params = best_params.copy()
                test_params[param_idx] = candidate
                
                # 使用预计算的相位图
                total_phase = base_phase.copy()
                for idx, value in enumerate(test_params):
                    if value != 0:  # 跳过零值参数
                        # 找到最接近的预计算值
                        closest_value = min(zernike_phases[idx].keys(), 
                                         key=lambda x: abs(x-value))
                        total_phase += zernike_phases[idx][closest_value]
                
                # 应用相位到SLM
                total_phase = np.mod(total_phase, 2 * np.pi)
                tophatscreen = (255 * total_phase / (2 * np.pi))
                tophat_screen_Corrected = IMGpy.SLM_screen_Correct(tophatscreen)
                slmx2.updateArray(tophat_screen_Corrected)
                time.sleep(0.5)
                # 评估成本函数
                cost_samples = []
                for _ in range(n_samples):
                    image=get_image()
                    image=image[1705:1825,1730:1850]

                    rotated_image = rotate(image,angle=34,reshape=False)
                    center=center_max_region(rotated_image)
                    rotated_image=rotated_image[int(center[1]-10):int(center[1]+11),int(center[0]-29):int(center[0]+30)]
                    
                    # center=dsigma_centroid(rotated_image)

                    # rotated_image=rotated_image[int(center[1]-4):int(center[1]+5),int(center[0]-23):int(center[0]+24)]

                    #center=[27,18]
                    
                    # h, w = image.shape
                    # radius = 3.5  # 圆半径

                    # Y, X = np.ogrid[:h, :w]
                    # dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

                    # mask = dist_from_center <= radius
                    # circle_pixels = image[mask]
                    # cost=1e5*circle_pixels.std()/circle_pixels.mean()
                    # image=image[int(center[1]-13):int(center[1]+14),int(center[0]-13):int(center[0]+14)]

                    # image=image[int(center[1]-13):int(center[1]+14),int(center[0]-13):int(center[0]+14)]
                    # cost=1e6*(1-abs(normalized_correlation(image,target)))
                    cost=1e6*(1-abs(calculate_ssim(win_size=3,target=target,actual=rotated_image)))
                    
                    cost_samples.append(cost)
                    time.sleep(0.01)
                avg_cost = filtered_mean(cost_samples)
                
                if avg_cost < current_best_cost:
                    current_best_cost = avg_cost
                    current_best_val = candidate
            
            if current_best_cost < best_cost:
                best_params[param_idx] = current_best_val
                best_cost = current_best_cost
                cost_improved = True
                print(f"Param {param_idx} updated to {current_best_val:.3f} | Cost: {best_cost:.4f}")
                plt.imshow(image,cmap='jet')
                plt.colorbar()
                plt.show()
                plot_3d(image)
        
        if not cost_improved and iter > 0:
            print(f"Early stopping at iteration {iter+1}")
            break
    
    return best_params, best_cost


#%%
ind_Zernike_phase1 = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# param_ranges=[(0.6,0.8)]+[(-0.1,0.1)]*6
# step_sizes=[0.01]*7
# param_ranges=[(-1,1)]*2+[(-0.2,0.2)]+[(-1,1)]+[(-0.2,0.2)]*12+[(-1,1)]*2
param_ranges=[(-0.2,0.2)]*18
step_sizes=[0.01]*18
#%%
#%%

zernike_phase = precalculate_zernike_phases(param_ranges, step_sizes, ind_Zernike_phase1)
#%%
best_params, best_cost=grid_search_zernike_optimized(
    param_ranges=param_ranges,
    zernike_phase=zernike_phase, 
    step_sizes=step_sizes, 
    max_iter=3,
    n_samples=20,
    early_stop_threshold=0.01
)
#%%
def optimized_grid_search(
    param_ranges,
    zernike_phase,
    step_sizes,
    max_iter=3,
    Flag=None,
    n_samples_list=None,
    best_params = None,
    shrink_n=0.3):
    """
    高级网格搜索：
    1. 每个参数可单独设置 n_samples。
    2. 第一轮后，对未更新的参数自动缩小搜索范围为最优值±shrink_n。
    """
    zernike_phases = zernike_phase
    n_params = len(param_ranges)

    if best_params is None:
        best_params = np.zeros(n_params)

    best_cost = float('inf')
    base_phase = np.load(path)
    # 每个参数的采样数
    if n_samples_list is None:
        n_samples_list = [5]*n_params
    current_ranges = list(param_ranges)
    unchanged_flags = [False]*n_params
    if Flag=='peak':
        phase_list = [np.load(path)] + [
        np.load(f'peak_only_8_{i}.npy') for i in range(n_params)
    ]
    for iter in range(max_iter):
        print(f"\n=== Iteration {iter+1}/{max_iter} ===")
        cost_improved = False
        updated_flags = [False]*n_params
        for param_idx in range(n_params):
            param_min, param_max = current_ranges[param_idx]
            step = step_sizes[param_idx]
            candidates = np.arange(param_min, param_max+step/2, step)
            n_samples = n_samples_list[param_idx] if param_idx < len(n_samples_list) else n_samples_list[-1]
            current_best_val = best_params[param_idx]
            current_best_cost = best_cost
            for candidate in tqdm(candidates, desc=f"Param {param_idx}"):
                test_params = best_params.copy()
                test_params[param_idx] = candidate
                total_phase = base_phase.copy()
                if Flag=='peak':

                # if 1-x1**2-x2**2-x3**2-x4**2<0:
                #     return 1e10
                # x0=np.sqrt(1-x1**2-x2**2-x3**2-x4**2)
                    exp_sum = np.exp(1j * phase_list[0])
                    for i in range(n_params):
                        exp_sum += test_params[i] * np.exp(1j * phase_list[i+1])
                    phase2 = np.angle(exp_sum)
                    total_phase = np.mod(phase2, 2 * np.pi)
                if Flag=='zernike':
                    for idx, value in enumerate(test_params):
                        if value != 0:
                            closest_value = min(zernike_phases[idx].keys(), key=lambda x: abs(x-value))
                            total_phase += zernike_phases[idx][closest_value]
                    total_phase = np.mod(total_phase, 2 * np.pi)

                tophatscreen = (255 * total_phase / (2 * np.pi))
                tophat_screen_Corrected = IMGpy.SLM_screen_Correct(tophatscreen)
                slmx2.updateArray(tophat_screen_Corrected)
                time.sleep(0.5)
                cost_samples = []
                for _ in range(n_samples):
                    image = get_image()
                    # image = image[1705-10:1825+10,1730-40:1850+10]
                    image = image[1725:1800,1730:1830]
                    # img=image.copy()
                    # max_val = img.max()
                    # threshold = max_val * 0.95
                    # img[img >= threshold] = threshold
                    # center = dsigma_centroid(image)
                    # h, w = image.shape
                    # radius = 3.5  # 圆半径
                    rotated_image = rotate(image,angle=34,reshape=False)
                    center=center_max_region(rotated_image)
                    # center=dsigma_centroid(rotated_image)
                    # rotated_image=rotated_image[int(center[1]-10):int(center[1]+11),int(center[0]-29):int(center[0]+30)]
                    # rotated_image=rotated_image[int(center[1]-13):int(center[1]+14),int(center[0]-35):int(center[0]+36)]
                    y=rotated_image[int(center[1]),int(center[0]-21):int(center[0]+22)]
                    # Y, X = np.ogrid[:h, :w]
                    # dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

                    # mask = dist_from_center <= radius
                    # circle_pixels = image[mask]
                    # cost=1e5*circle_pixels.std()/circle_pixels.mean()

                    # image=image[int(center[1]-12):int(center[1]+15),int(center[0]-12):int(center[0]+15)]
                    
                    # image = image[int(center[1]-13):int(center[1]+14),int(center[0]-13):int(center[0]+14)]
                    # x=rotated_image[int(center[1]),int(center[0]-21):int(center[0]+22)]
                    # y=rotated_image[int(center[1]+1),int(center[0]-21):int(center[0]+22)]
                    # z=rotated_image[int(center[1]-1),int(center[0]-21):int(center[0]+22)]
                    # cost =max(x.std()/x.mean(),y.std()/y.mean(),z.std()/z.mean())


                    if all(r >= t for r, t in zip(rotated_image.shape, target.shape)):
                        cost=100*(y.std()/y.mean())
                        # cost=1e6*(1-abs(sliding_max_ncc(target[3:-3,8:-8],rotated_image[3:-3,8:-8])))
                        # cost=1e6*(1-abs(normalized_correlation(rotated_image,target)))
                        # cost = 1e6*(1-abs(calculate_ssim(win_size=3,target=target,actual=rotated_image)))

                        cost_samples.append(cost)
                    time.sleep(0.01)
                avg_cost = hist_mean(cost_samples)
                if np.isnan(hist_mean(cost_samples)):
                    avg_cost=1e10
                if avg_cost < current_best_cost:
                    current_best_cost = avg_cost
                    current_best_val = candidate
                    current_best_image=rotated_image.copy()
                    current_best_hist=cost_samples.copy()
                    current_best_phase = total_phase
            if current_best_cost < best_cost:
                best_params[param_idx] = current_best_val
                best_cost = current_best_cost
                cost_improved = True
                print(f"Param {param_idx} updated to {current_best_val:.3f} | Cost: {best_cost:.4f}")
                # load_zernike_phase_on_slm(best_params,zernike_phase)
                # time.sleep(0.5)
                # image = get_image()
                # image = image[1725:1800,1730:1830]
                # rotated_image = rotate(image,angle=34,reshape=False)
                # center=center_max_region(rotated_image)
                # rotated_image=rotated_image[int(center[1]-10):int(center[1]+11),int(center[0]-29):int(center[0]+30)]
                plt.imshow( current_best_image,cmap='jet')
                plt.show()
                plot_3d( current_best_image)
                print(1e6*(1-abs(sliding_max_ncc(target[3:-3,8:-8],rotated_image[3:-3,8:-8]))))
                plt.hist(current_best_hist)
                plt.show()
                updated_flags[param_idx] = True
        # 第一轮后，所有参数收缩范围
        if iter == 0:
            for i in range(n_params):
                center = best_params[i]
                orig_min, orig_max = param_ranges[i]
                new_min = max(center - shrink_n, orig_min)
                new_max = min(center + shrink_n, orig_max)
                current_ranges[i] = (new_min, new_max)
        if not cost_improved and iter > 0:
            print(f"Early stopping at iteration {iter+1}")
            break
    return best_params, best_cost,current_best_image,current_best_phase
#%%
ind_Zernike_phase1 = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
param_ranges=[(-0.5,0.5)]*18
step_sizes=[0.01]*18
Flag='zernike'
#%%
ind_Zernike_phase1 = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
param_ranges=[(-0.002,0.002)]*8
step_sizes=[0.0001]*8
Flag='peak'
#%%
best_params, best_cost,current_best_image,current_best_phase=optimized_grid_search(
    param_ranges=param_ranges,
    zernike_phase=zernike_phase2, 
    step_sizes=step_sizes, 
    max_iter=8,
    n_samples_list=[30]+[30]*17,
    Flag=Flag,
    shrink_n=0.02
)
print(best_params)
#%%
tophatscreen=(255*current_best_phase/(2*np.pi))
# tophatscreen=(255*generate_tilted_grating([1024,1272],angle_deg=0)/(2*np.pi))

# tophatscreen=(255*phase/(2*np.pi))
# tophatscreen=255*np.array(cg1.slm_phase_end)/(2*np.pi)
tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
# tophat_screen_Corrected=(215*tophatscreen/255).astype('uint8')
slmx2.updateArray(tophat_screen_Corrected)

#%%
load_zernike_phase_on_slm(best_params, zernike_phase2)

#%%
add_zernike_phase(best_params,ind_Zernike_phase1)# %%
#%%
total_phase = np.load(path)
for idx, value in enumerate(best_params):
    if value != 0:
        closest_value = min(zernike_phase[idx].keys(), key=lambda x: abs(x-value))
        total_phase += zernike_phase[idx][closest_value]
total_phase = np.mod(total_phase, 2 * np.pi)
np.save('phaseold_wcg=100_guassianx=30_guassiany=20_ir5.npy',total_phase)
#%%
N=[1024,1272]
def fresnel_lens_phase_generate(shift_distance, SLMRes=(1024,1272), x0=N[0]/2, y0=N[1]/2, pixelpitch=12.5,wavelength=0.813,focallength=200000,magnification=1):

        Xps, Yps = torch.meshgrid(torch.linspace(0, SLMRes[0], SLMRes[0]), torch.linspace(0, SLMRes[1], SLMRes[1]))


        X = (Xps-x0)*pixelpitch
        Y = (Yps-y0)*pixelpitch

        fresnel_lens_phase = torch.fmod(torch.pi*(X**2+Y**2)*shift_distance/(wavelength*focallength**2)*magnification**2,2*torch.pi)

        return np.array(torch.remainder(fresnel_lens_phase,2*torch.pi))
# %%
def add_fresnel(x):
    fresnel_lens_phase = fresnel_lens_phase_generate(x)
    phase = np.mod(np.load(path) + fresnel_lens_phase, 2 * np.pi)
    tophatscreen = (255 * phase / (2 * np.pi))
    tophat_screen_Corrected = IMGpy.SLM_screen_Correct(tophatscreen)
    slmx2.updateArray(tophat_screen_Corrected)
# %%
for i in range(20):
    add_zernike_phase([0.1*(i-10)],[7])
    image=get_image()
    image=image[1750:1785,1770:1820]

    plt.imshow(image,cmap='jet')
    print(image.max())
    plot_3d(image)


#%%
def get_pupil_radius_from_L(path='L.npy', threshold_ratio=0.2, show=True):
    """
    通过L.npy光强分布自动确定光瞳口径（等效半径）
    :param path: L.npy路径
    :param threshold_ratio: 阈值比例（如0.2表示最大值的20%）
    :param show: 是否显示可视化
    :return: 光瞳半径、中心、掩码
    """
    L = np.load(path)
    L = L.astype(np.float64)
    L -= L.min()
    L /= (L.max() + 1e-8)

    # 阈值分割
    mask = L > threshold_ratio

    # 计算质心
    y, x = np.indices(L.shape)
    total = mask.sum()
    cx = (x * mask).sum() / total
    cy = (y * mask).sum() / total

    # 计算等效半径
    r = np.sqrt(mask.sum() / np.pi)

    if show:
        plt.imshow(L, cmap='jet')
        plt.contour(mask, colors='w')
        circle = plt.Circle((cx, cy), r, color='r', fill=False, linewidth=2)
        plt.gca().add_patch(circle)
        plt.title(f'光瞳中心: ({cx:.1f}, {cy:.1f})  半径: {r:.1f}')
        plt.show()

    return r, (cx, cy), mask

# 用法
radius, center, pupil_mask = get_pupil_radius_from_L('L.npy', threshold_ratio=0.2)
print(f"光瞳半径: {radius:.2f}, 中心: {center}")
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
    
def fit_circle_least_squares(points):
    # 拟合圆: (x-a)^2 + (y-b)^2 = r^2
    x = points[:, 0]
    y = points[:, 1]
    A = np.c_[2*x, 2*y, np.ones_like(x)]
    b = x**2 + y**2
    c, resid, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b, d = c
    center = (a, b)
    radius = np.sqrt(d + a**2 + b**2)
    return center, radius
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
X,Y=np.meshgrid(np.linspace(1,SLMResX,SLMResX),np.linspace(1,SLMResY,SLMResY))
X=X-SLMResX/2
Y=Y-SLMResY/2
r=np.sqrt(X**2+Y**2)

theta=np.mod(np.arctan2(Y,X),2*np.pi)

def Phase_LG(p,l):
    return np.mod(-l*theta+np.pi*np.heaviside(-genlaguerre(p,np.abs(l))(2*r**2/w0**2),1),2*np.pi)

p=1
l=2
plt.imshow(Phase_LG(p,l))
plt.colorbar()
plt.show()
#%%
slm_phase=Phase_LG(p,l)
tophatscreen=(255*slm_phase/(2*np.pi))
tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
slmx2.updateArray(tophat_screen_Corrected)
time.sleep(1)
images=get_image()
images=images[1680:1820-60,1780+75:1930]
images = rotate(images,angle=12,reshape=False)
image=np.flip(images, axis=0)

plt.imshow(image)
print(image.max())
#%%
img = image.astype(np.float64)
img -= img.min()
img /= (img.max() + 1e-8)
edges = feature.canny(img, sigma=5) 
y_idx, x_idx = np.nonzero(edges)
points = np.column_stack((x_idx, y_idx))
center, radius = fit_circle_least_squares(points)
print("拟合圆心:", center, "拟合半径:", radius)
import matplotlib.pyplot as plt

plt.imshow(image, cmap='gray')
circle = plt.Circle(center, radius, color='r', fill=False, linewidth=2)
plt.gca().add_patch(circle)
plt.title(f'拟合半径: {radius:.1f}')
plt.show()

image_target=np.sqrt(image[int(center[1]-40):int(center[1]+41),int(center[0]-40):int(center[0]+41)])
# %%
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
E_out_p,E_out_amp,E_out_2=dFFT(slm_phase)
img = np.array(E_out_2[3475:3550,3475:3550]).astype(np.float64)
img -= img.min()
img /= (img.max() + 1e-8)
edges = feature.canny(img, sigma=5) 
y_idx, x_idx = np.nonzero(edges)
points = np.column_stack((x_idx, y_idx))
center, radius = fit_circle_least_squares(points)
print("拟合圆心:", center, "拟合半径:", radius)
import matplotlib.pyplot as plt
plt.imshow(E_out_2[3475:3550,3475:3550], cmap='gray')
circle = plt.Circle(center, radius, color='r', fill=False, linewidth=2)
plt.gca().add_patch(circle)
plt.title(f'拟合半径: {radius:.1f}')
plt.show()
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
#%%
def gs_vortex_correction(slm_phase,image_target, max_iter=100, tol=0.01, show=False):

    E_out_p,E_out_amp_init=dFFT(slm_phase)
    centerx=d4sigma_centroid(E_out_amp_init)
    imagex=E_out_amp_init.clone()
    E_out_amp_init_cut=E_out_amp_init[int(centerx[1]-40):int(centerx[1]+41),int(centerx[0]-40):int(centerx[0]+41)]
    image_target=torch.tensor(image_target.copy())*torch.sqrt((E_out_amp_init_cut**2).sum()/((image_target**2).sum()))
    imagex[int(centerx[1]-40):int(centerx[1]+41),int(centerx[0]-40):int(centerx[0]+41)]=image_target
    pre_error=np.inf

    for i in range(max_iter):
        E_in_p,E_in_amp=dIFFT(imagex,E_out_p)
        E_out_p,E_out_amp=dFFT(E_in_p)
        error = np.linalg.norm(E_out_amp - E_out_amp_init) / np.linalg.norm(E_out_amp_init)
        if show and i % 1 == 0:
            print(f'Iter {i}, error={error:.4f}')
        if np.abs( error-pre_error) < tol:
            print(f'Converged at iter {i}, error={error:.4f}')
            break
        pre_error=error

    return -(np.array(E_in_p)-slm_phase)

# 用法示例
# target_intensity = ... # 你的目标圆环强度分布
# aperture_mask = ...    # SLM口径掩码
# H, C = gs_vortex_correction(target_intensity, aperture_mask, l=1, max_iter=100, tol=0.01, show=True)
#%%
update_phase=slm_phase
correcti=[]
# %%
for i in range(5):
    correct=gs_vortex_correction(slm_phase,image_target, max_iter=100, tol=0.0005, show=True)
    correcti.append(correct)
    update_phase=update_phase+correct
    tophatscreen=(255*np.mod((update_phase),2*np.pi)/(2*np.pi))
    tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
    slmx2.updateArray(tophat_screen_Corrected)
    time.sleep(0.5)
    images=get_image()
    images=images[1680:1820,1780:1880]
    images= rotate(images,angle=34,reshape=False)
    image=np.flip(images, axis=0)

    plt.imshow(image)
    plt.show()
    print(image.max())
    img = image.astype(np.float64)
    img -= img.min()
    img /= (img.max() + 1e-8)
    edges = feature.canny(img, sigma=5) 
    y_idx, x_idx = np.nonzero(edges)
    points = np.column_stack((x_idx, y_idx))
    center, radius = fit_circle_least_squares(points)
    print("拟合圆心:", center, "拟合半径:", radius)
    import matplotlib.pyplot as plt

    plt.imshow(image, cmap='gray')
    circle = plt.Circle(center, radius, color='r', fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    plt.title(f'拟合半径: {radius:.1f}')
    plt.show()

    image_target=np.sqrt(image[int(center[1]-40):int(center[1]+41),int(center[0]-40):int(center[0]+41)])
# %%
def sigma_mean_img(img_list):
    img_stack = np.dstack(img_list)
    
    # 使用中值法计算每个像素的基础值
    median_img = np.median(img_stack, axis=2)
    
    # 计算每个像素与中值的偏差
    diff_stack = np.abs(img_stack - median_img[:, :, np.newaxis])
    
    # 计算每个像素的鲁棒平均（排除离群值）
    sigma = 1 # 标准差阈值
    avg_image = np.zeros_like(img_list[0], dtype=np.float32)
    pixel_count = np.zeros_like(img_list[0], dtype=np.float32)
    
    for i in range(img_stack.shape[2]):
        # 创建每个像素的掩码（排除异常值）
        mask = diff_stack[:, :, i] < sigma * np.std(img_stack, axis=2)
        avg_image += np.where(mask, img_stack[:, :, i], 0)
        pixel_count += mask.astype(np.float32)
    
    # 防止除零错误
    pixel_count[pixel_count == 0] = 1
    avg_image /= pixel_count
    return avg_image
#%%
rep=30
slm_phase=np.load(path)
tophatscreen=(255*slm_phase/(2*np.pi))
tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
slmx2.updateArray(tophat_screen_Corrected)
time.sleep(0.5)
x=31
y=71
#%%
images=get_image()
images=images[1000:1830,1780:1920]
# images=images[1710:1800,1780:1850]
images = rotate(images,angle=12,reshape=False)
image=np.flip(images, axis=0)
# plt.imshow(image)
# print(image.max())
center=center_max_region(image)
image_target=np.sqrt(image[int(center[1]-x//2):int(center[1]+x//2+1),int(center[0]-y//2):int(center[0]+y//2+1)])
#%%
img_list=[]
for i in range(rep):
    images=get_image()
    # images=images[1650:1830,1780:1920]
    images=images[1000:1830,1780:1920]
    images = rotate(images,angle=12,reshape=False)
    image=np.flip(images, axis=0)
    
# plt.imshow(image)
# print(image.max())
    center=center_max_region(image)
    img_list.append(image.copy()[round(center[1]-x//2):round(center[1]+x//2+1),round(center[0]-y//2):round(center[0]+y//2+1)])
    time.sleep(0.1)
# image_target=np.sqrt(image[int(center[1]-20):int(center[1]+21),int(center[0]-20):int(center[0]+21)])
avg_image = np.zeros_like(img_list[0], dtype=np.float32)
for img in img_list:
    avg_image += img.astype(np.float32) / rep
image_target=np.sqrt(avg_image)
# image_target=avg_image

#%%
# def gs_vortex_correction(slm_phase,image_target, max_iter=100, tol=0.01, show=False):

#     E_out_p,E_out_amp_init=dFFT(slm_phase)
#     centerx=d4sigma_centroid(E_out_amp_init)
#     imagex=E_out_amp_init.clone()
#     E_out_amp_init_cut=E_out_amp_init[int(centerx[1]-13):int(centerx[1]+14),int(centerx[0]-35):int(centerx[0]+36)]
#     image_target=torch.tensor(image_target.copy())*torch.sqrt((E_out_amp_init_cut**2).sum()/((image_target**2).sum()))
#     imagex[int(centerx[1]-13):int(centerx[1]+14),int(centerx[0]-35):int(centerx[0]+36)]=image_target
    
#     pre_error=np.inf
#     error_history = []
#     min_error = np.inf
#     best_phase = slm_phase
#     elbow_detected = False
#     stagnation_count = 0

#     # 肘部法则参数
#     ELBOW_THRESHOLD = 0.1  # 下降率阈值
#     STAGNATION_LIMIT = 3   # 连续停滞次数
#     MIN_ITER = 5           # 最小迭代次数
#     for i in range(max_iter):
#         E_in_p,E_in_amp=dIFFT(imagex,E_out_p)
#         E_out_p,E_out_amp=dFFT(E_in_p)
#         current_error  = np.linalg.norm(E_out_amp - E_out_amp_init) / np.linalg.norm(E_out_amp_init)
#         error_history.append(current_error)

#         if current_error < min_error:
#             min_error = current_error
#             best_phase = E_in_p.clone()

#         if i >= MIN_ITER:
#             recent_errors = error_history[-5:]
#             improvement_rates = []
#             for j in range(1, len(recent_errors)):
#                 rate = (recent_errors[j-1] - recent_errors[j]) / recent_errors[j-1]
#                 improvement_rates.append(rate)
#                 for j in range(1, len(recent_errors)):
#                     rate = (recent_errors[j-1] - recent_errors[j]) / recent_errors[j-1]
#                     improvement_rates.append(rate)
        
#                 avg_improvement = np.mean(improvement_rates)

#                 if avg_improvement < ELBOW_THRESHOLD and not elbow_detected:
#                     print(f"Elbow detected at iter {i} with improvement rate {avg_improvement:.4f}")
#                     elbow_detected = True

#                 if np.abs(current_error - prev_error) < tol * prev_error:
#                     stagnation_count += 1
#                 else:
#                     stagnation_count = 0
#         if (elbow_detected and stagnation_count >= 1) or stagnation_count >= STAGNATION_LIMIT:
#             print(f"Terminated at iter {i} due to stagnation after elbow point")
#             E_in_p = best_phase  # 恢复最佳相位
#             break
#         prev_error = current_error
    
#         if show and i % 1 == 0:
#             print(f'Iter {i}, error={current_error:.4f}')


#     return -(np.array(E_in_p)-slm_phase)

# def gs_vortex_correction(slm_phase,image_target, max_iter=100, tol=0.01, show=False):

#     E_out_p,E_out_amp_init=dFFT(slm_phase)
#     centerx=d4sigma_centroid(E_out_amp_init)
#     imagex=E_out_amp_init.clone()
#     E_out_amp_init_cut=E_out_amp_init[int(centerx[1]-13):int(centerx[1]+14),int(centerx[0]-35):int(centerx[0]+36)]
#     image_target=torch.tensor(image_target.copy())*torch.sqrt((E_out_amp_init_cut**2).sum()/((image_target**2).sum()))
#     imagex[int(centerx[1]-13):int(centerx[1]+14),int(centerx[0]-35):int(centerx[0]+36)]=image_target
#     pre_error=np.inf

#     for i in range(max_iter):
#         E_in_p,E_in_amp=dIFFT(imagex,E_out_p)
#         E_out_p,E_out_amp=dFFT(E_in_p)
#         error = np.linalg.norm(E_out_amp - E_out_amp_init) / np.linalg.norm(E_out_amp_init)
#         if show and i % 1 == 0:
#             print(f'Iter {i}, error={error:.4f}')
#         if np.abs( error-pre_error) < tol*error:
#             print(f'Converged at iter {i}, error={error:.4f}')
#             break
#         pre_error=error
#         # error = np.linalg.norm(E_out_amp - imagex) / np.linalg.norm(imagex)
#         # if show and i % 1 == 0:
#         #     print(f'Iter {i}, error={error:.4f}')
#         # if np.abs( error )< tol:
#         #     print(f'Converged at iter {i}, error={error:.4f}')
#         #     break


#     return -(np.array(E_in_p)-slm_phase)
# # def gs_vortex_correction(slm_phase,image_target, max_iter=100, tol=0.01, show=False):
#     image_target=torch.tensor(image_target.copy())
#     E_out_p,E_out_amp_init=dFFT(slm_phase)
#     centerx=d4sigma_centroid(E_out_amp_init)
#     imagex = torch.zeros_like(E_out_amp_init)
#     target_region = E_out_amp_init[int(centerx[1]-13):int(centerx[1]+14),int(centerx[0]-35):int(centerx[0]+36)]
#     scale = torch.sqrt(torch.sum(target_region**2) / torch.sum(image_target**2))
#     norm_target = image_target * scale
#     imagex[int(centerx[1]-13):int(centerx[1]+14),int(centerx[0]-35):int(centerx[0]+36)] = norm_target
#     pre_error=np.inf

#     for i in range(max_iter):
#         E_in_p,E_in_amp=dIFFT(imagex,E_out_p)
#         E_out_p,E_out_amp=dFFT(E_in_p)
#         region = E_out_amp[int(centerx[1]-13):int(centerx[1]+14),int(centerx[0]-35):int(centerx[0]+36)]
#         error = np.linalg.norm(region - norm_target) / np.linalg.norm(norm_target)
#         if show and i % 1 == 0:
#             print(f'Iter {i}, error={error:.4f}')
#         if np.abs( error-pre_error) < tol:
#             print(f'Converged at iter {i}, error={error:.4f}')
#             break
#         pre_error=error

#     return -(np.array(E_in_p)-slm_phase)
# # def gs_vortex_correction(slm_phase,image_target, max_iter=100, tol=0.01, show=False):

#     E_out_p,E_out_amp_init=dFFT(slm_phase)
#     centerx=d4sigma_centroid(E_out_amp_init)
#     imagex=E_out_amp_init.clone()
#     E_out_amp_init_cut=E_out_amp_init[int(centerx[1]-20):int(centerx[1]+21),int(centerx[0]-20):int(centerx[0]+21)]
#     image_target=torch.tensor(image_target.copy())*torch.sqrt((E_out_amp_init_cut**2).sum()/((image_target**2).sum()))
#     imagex[int(centerx[1]-20):int(centerx[1]+21),int(centerx[0]-20):int(centerx[0]+21)]=image_target
#     pre_error=np.inf

#     for i in range(max_iter):
#         E_in_p,E_in_amp=dIFFT(imagex,E_out_p)
#         E_out_p,E_out_amp=dFFT(E_in_p)
#         error = np.linalg.norm(E_out_amp - E_out_amp_init) / np.linalg.norm(E_out_amp_init)
#         if show and i % 1 == 0:
#             print(f'Iter {i}, error={error:.4f}')
#         if np.abs( error-pre_error) < tol:
#             print(f'Converged at iter {i}, error={error:.4f}')
#             break
#         pre_error=error

#     return -(np.array(E_in_p)-slm_phase)
#%%
update_phase=slm_phase
correct_sum=[]
z_list=[]
#%%
for i in range(10):
    correct=gs_vortex_correction(slm_phase,image_target, max_iter=8, tol=0.0003, show=True)
    # correct=phase_retrieval_spot(image_target,lr=0.5, rep=200)
    # correct=gs_zeropad_correction(slm_phase,image_target, max_iter=5, tol=0.0006, show=True)

    # correct=HIO_correction(slm_phase,image_target, max_iter=60, tol=0.0006, show=True)
    correct_sum.append(correct)
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
        images=images[1000:1830,1780:1920]
        # images=images[1650:1830,1780:1920]
        images = rotate(images,angle=12,reshape=False)
        image=np.flip(images, axis=0)
    # plt.imshow(image)
    # print(image.max())
        center=center_max_region(image)
        img_list.append(image.copy()[round(center[1]-x//2):round(center[1]+x//2+1),round(center[0]-y//2):round(center[0]+y//2+1)])
        time.sleep(0.1)
    # image_target=np.sqrt(image[int(center[1]-20):int(center[1]+21),int(center[0]-20):int(center[0]+21)])
    avg_image = np.zeros_like(img_list[0], dtype=np.float32)
    for img in img_list:
        avg_image += img.astype(np.float32) / rep
    plot_3d(avg_image)
    # plt.imshow(avg_image,cmap='jet')
    # plt.show()
    # z=avg_image[int(x//2),int(y//2-20):int(y//2+21)]
    # print(100*z.std()/z.mean())
    # print(100*(z.max()-z.min())/z.mean())
    # z_list.append(z.std()/z.mean())
    # plt.plot(z)
    # plt.show()
    # if z.std()/z.mean()-min(z_list)<=1e-6:
    #     best_phase=update_phase
    #     best_cost=z.std()/z.mean()
    image_target=np.sqrt(avg_image)
    # image_target=avg_image

# %%
slm_phase=np.load(path)
tophatscreen=(255*np.mod(slm_phase+correct_sum[0],2*np.pi)/(2*np.pi))
tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
slmx2.updateArray(tophat_screen_Corrected)
time.sleep(1)
test_image()
# %%
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


def gs_zeropad_correction(slm_phase,image_target, max_iter=100, tol=0.01, show=False):
    x,y=image_target.shape
    E_out_p,E_out_amp_init=dFFT(slm_phase)
    centerx=d4sigma_centroid(E_out_amp_init)
    imagex=torch.zeros_like(E_out_amp_init)
    E_out_amp_init_cut=E_out_amp_init[int(centerx[1]-x//2):int(centerx[1]+x//2+1),int(centerx[0]-y//2):int(centerx[0]+y//2+1)]
    image_target=torch.tensor(image_target.copy())*torch.sqrt((E_out_amp_init_cut**2).sum()/((image_target**2).sum()))
    imagex[int(centerx[1]-x//2):int(centerx[1]+x//2+1),int(centerx[0]-y//2):int(centerx[0]+y//2+1)]=image_target
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
# %%
def HIO_correction(slm_phase,image_target,beta=0.8, max_iter=100, tol=0.01, show=False):
    x,y=image_target.shape
    E_out_p,E_out_amp_init=dFFT(slm_phase)
    centerx=d4sigma_centroid(E_out_amp_init)
    imagex=E_out_amp_init.clone()
    E_out_amp_init_cut=E_out_amp_init[int(centerx[1]-x//2):int(centerx[1]+x//2+1),int(centerx[0]-y//2):int(centerx[0]+y//2+1)]
    image_target=torch.tensor(image_target.copy())*torch.sqrt((E_out_amp_init_cut**2).sum()/((image_target**2).sum()))
    imagex[int(centerx[1]-x//2):int(centerx[1]+x//2+1),int(centerx[0]-y//2):int(centerx[0]+y//2+1)]=image_target
    pre_error=np.inf
    best_phase = slm_phase.copy()
    best_error = np.inf
    new_phase= slm_phase.copy()

    for i in range(max_iter):
        E_in_p,E_in_amp=dIFFT(imagex,E_out_p)
        constraint = E_in_amp > 0.1*E_in_amp.max()
        new_phase = torch.where(
            constraint,
            E_in_p,  
            torch.tensor(new_phase) - beta * E_in_p  
        )
        E_out_p,E_out_amp=dFFT(new_phase)
        error = np.linalg.norm(E_out_amp - imagex) / np.linalg.norm(imagex)
        if error < best_error:
            best_error = error
            best_phase = new_phase.clone()
        if show and i % 1 == 0:
            print(f'Iter {i}, error={error:.4f}')
        if np.abs( error-pre_error) < tol:
            print(f'Converged at iter {i}, error={error:.4f}')
            break
        pre_error=error

    return -(np.array(best_phase)-slm_phase)
# %%
def phase_retrieval_spot(image_I,lr=0.1, rep=1000):
    x,y = image_I.shape
    E_out_p,E_out_amp=dFFT(slm_phase)
    centerx=d4sigma_centroid(E_out_amp)
    total=(E_out_amp**2).sum()
    phase=slm_phase.copy()
    for i in range(rep):
        E_out_p,E_out_amp=dFFT(phase)
        E_out_amp_cut=E_out_amp[int(centerx[1]-x//2):int(centerx[1]+x//2+1),int(centerx[0]-y//2):int(centerx[0]+y//2+1)]
        image_I=image_I*(E_out_amp_cut**2).sum().item()/image_I.sum()
        diff = (E_out_amp_cut**2 - image_I)
        loss = 0.5 * (diff**2).sum()
        grad_y=2*diff*E_out_amp_cut
        E_out_amp_fix=torch.zeros_like(E_out_amp)
        E_out_amp_fix[int(centerx[1]-x//2):int(centerx[1]+x//2+1),int(centerx[0]-y//2):int(centerx[0]+y//2+1)]=grad_y
        # E_out_amp=E_out_amp*torch.sqrt(total/(E_out_amp**2).sum())
        E_in_p,E_in_amp=dIFFT(E_out_amp_fix,E_out_p)
        grad_theta=np.imag(np.conj(L*np.exp(1j*phase))*E_in_amp*np.exp(1j*E_in_p))
        phase-=lr*np.array(grad_theta)
        if i % 10 == 0 or i == rep-1:
            print(f"Epoch {i}/{rep}, Loss: {loss:.4e}")
            print(np.linalg.norm(grad_theta))
    return -(np.array(phase)-slm_phase)
# %%
