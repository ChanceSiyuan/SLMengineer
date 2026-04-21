#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import fft
import matplotlib.pyplot as plt
from PIL import Image
import os
import IMGpy
import slmpy
import torch
import time
from scipy.ndimage import binary_dilation, rotate
import pickle  
from scipy.optimize import curve_fit
from skimage import restoration
from skimage.restoration import inpaint
from skimage.measure import label
from scipy.ndimage import generic_filter
from skimage.restoration import unwrap_phase
import cv2

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
# %%
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

## 以上代码是海康相机的封装，下面为运行代码。
#############################################

#%%
enum_devices()
open_device(0)
# reset_roi()

# %%
start_grabbing()
image=get_image()
plt.imshow(image)
print(image.max())

#%%
image=get_image()
# image=image[1200:1600,3300:3700]
# image=image[1350:1800,2450:2890]
plt.imshow(image)
print(image.max())



#%%
slmheight = 1024    # SLM垂直分辨率
slmwidth = 1272     # SLM水平分辨率
size=[1024,1272]
def generate_raster_matrix(size, period = 8):
    x_coords = np.arange(1, size[1] + 1)  
    phase_values = x_coords % period       

    a = np.tile(phase_values, (size[0], 1))
    a = (a / (period - 1)) * 2*np.pi  
    a = a
    return a

def generate_tilted_grating(size, period = 4,angle_deg=45):
    height, width = size[0], size[1]
    x_coords = (np.arange(width) - width / 2)
    y_coords = (np.arange(height) - height / 2)
    X, Y = np.meshgrid(x_coords, y_coords)

    angle_rad = np.deg2rad(angle_deg)
    X_rot = X * np.cos(angle_rad) + Y * np.sin(angle_rad)

    phase = np.mod(2 * np.pi * X_rot / period, 2 * np.pi)

    return phase

def replace_block_padded(arr, i, j, block):
    """
    在对称填充后的数组中替换指定坐标的34x34小块
    :param arr: 填充后的数组 (1024, 1272)
    :param i: 行坐标 (0 ≤ i < 30)
    :param j: 列坐标 (0 ≤ j < 37)
    :param block: 待替换的34x34数组
    :return: 修改后的填充数组
    """
    m, n = block.shape
    # 计算填充偏移量
    pad_top = (arr.shape[0] - original_shape[0]) // 2  # 上下对称填充，取整
    pad_left = (arr.shape[1] - original_shape[1]) // 2  # 左右对称填充，取整
    
    # 计算原始数据区域内的切片范围
    row_start = pad_top + i * m
    row_end = pad_top + (i + 1) * m
    col_start = pad_left + j * n
    col_end = pad_left + (j + 1) * n
    # 验证是否超出原始数据区域
    
    # 执行替换
    new_arr=np.copy(arr)
    new_arr[row_start:row_end, col_start:col_end] = block
    return new_arr

def measure_amp(rep=10):  # rep指连续拍照次数，因此里面的函数要自己改。
    list=[]
    for i in range(0, 1024//n):
    # 为当前i创建一个独立的数据存储结构
        current_i_lists = []
        for j in range(1272 // m):
            sumx=[]
            print(f"Processing i={i}, j={j}")
            padded_array = 255*replace_block_padded(array_x, i, j, grating[0])/(2*np.pi)
            SLM_screen_Corrected=np.mod(padded_array,256)
            phase_to_screen=np.around(SLM_screen_Corrected*215/255).astype('uint8')
            # phase_to_screen = (215 * (padded_array / (2 * np.pi))).astype(np.uint8)
            slm.updateArray(phase_to_screen)
            time.sleep(0.5)
            for k in range(rep):
                image=get_image()
                image=image[1200:1600,3300:3700]
                if image.max()>4090:
                    print("guobao")
                    return None
                sumx.append(image.sum())
                time.sleep(0.05)
            current_i_lists.append(filtered_mean(sumx))      
        list.append(current_i_lists)
    return list

def amp_get(lists):
    listx=[]
    for i in range(1024//n):
    # 为当前i创建一个独立的数据存储结构
        current_i_lists = []
        for j in range(1272 // m):
                print(f"Processing i={i}, j={j}")
                current_i_lists.append(float((max(lists[i][j]-51657.028169014084,0))))   # 硬编码的背景噪声求和
        listx.append(current_i_lists)
    listx=np.array(listx)
    return listx

def plot_3d(image):
    rows, cols = image.shape

    # 创建 X 和 Y 坐标
    x = np.arange(cols)  # 列索引为 X 坐标
    y = np.arange(rows)  # 行索引为 Y 坐标
    X, Y = np.meshgrid(x, y)
    Z = image

    # 创建 3D 图形
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    ax.set_title('3D Visualization of rotated_image')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

def filtered_mean(data):
    """
    计算过滤掉均值±标准差范围外数据后的平均值
    """
    arr = np.array(data, dtype=np.float64)
    
    # 计算均值和标准差（默认使用总体标准差ddof=0）
    mean = np.mean(arr)
    std = np.std(arr)
    
    # 确定过滤范围 [mean - std, mean + std]
    lower_bound = mean - std
    upper_bound = mean + std
    
    # 过滤数据（保留范围内的值）
    filtered = arr[(arr >= lower_bound) & (arr <= upper_bound)]
    if len(filtered) == 0:
        print("警告：过滤后数据为空！")
        return None
    return np.mean(filtered)

def calculate_centroid(image):
    """
    使用质心法计算图像中光斑的中心坐标
    :param image: 输入图像（需为灰度图）
    :return: (cx, cy) 质心坐标
    """
    image = image.astype(np.float64)
    image[image<40]=0
    height, width = image.shape
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    
    total_intensity = np.sum(image)
    if total_intensity == 0:
        raise ValueError("图像中未检测到光斑")
    cx = np.sum(xx * image) / total_intensity
    cy = np.sum(yy * image) / total_intensity
    return (cx, cy)


#%%
# --- 1. 扫描参数与几何尺寸设置 ---
n, m = 32, 32  # 定义扫描的基本单元（小块）尺寸，即在 SLM 上每 32x32 像素作为一个采样点
# 计算 SLM 能够整除小块后的原始有效区域尺寸（1024x1248），去除无法凑成整块的余数部分
original_shape = (1024 - 1024 % n, 1272 - 1272 % m)
target_shape = (1024, 1272)  # SLM 的全屏分辨率

s = 3  # 相移步数（Phase Steps），通常使用 3 步法计算相位：0, 2π/3, 4π/3
# 计算 SLM 屏幕的正中心块索引，用于后续作为参考点或对准
# 这里强行将固定块放在SLM的正中心
center = np.round([((1024 // n) - 1) / 2, ((1272 // m) - 1) / 2]).astype(int)

# --- 2. 衍射光栅生成 ---
# 生成 s 组具有特定相移的衍射光栅（用于将光偏转到第一级衍射，从而进行相位提取），2pi整除步数s。
# 首先尝试生成光栅矩阵（Raster Matrix），随后被下方的斜光栅（Tilted Grating）覆盖
grating = [
    np.mod(generate_raster_matrix([n, m]) + i * (2 * np.pi / s), 2 * np.pi)
    for i in range(s)
]
# 生成 3 组带有 120 度等间距相移的斜光栅（这是实际用于相移干涉扫描的单元）
grating = [
    np.mod(generate_tilted_grating([n, m]) + i * (2 * np.pi / s), 2 * np.pi)
    for i in range(s)
]

# 初始化全屏零相位矩阵（背景）
array_x = np.zeros((1024, 1272))
# 在 SLM 中心位置放置一个带相移的光栅块，用于初步测试
array = replace_block_padded(array_x, center[0], center[1], grating[0])
# 生成一个掩膜（Mask），仅在中心块区域为 1，其余为 0，用于提取该区域的校正值
ind = replace_block_padded(array_x, center[0], center[1], np.ones((n, m)))
# 加载 SLM 厂方提供的波前校正文件（通常为 810nm 下的基准平整度校正图）
correction = np.array(Image.open('CAL_LSH0803519_810nm.bmp'))

# --- 3. SLM 设备初始化 ---
# 连接 SLM 控制器，monitor=0 通常指代主显示器之外的扩展屏
slm = slmpy.SLMdisplay(monitor=0, isImageLock = True)

# --- 4. 局部块扫描测试 (单元验证) ---
i = 0  # 测试扫描第 0 行
j = 0  # 测试扫描第 0 列
# 生成当前扫描块的掩膜
padded_ind = replace_block_padded(ind, i, j, np.ones((n, m)))
# 将 0-2π 的相位值映射到 0-255 的灰度级，并放置在 SLM 的第 (i, j) 块位置
padded_array = 255 * replace_block_padded(array, i, j, grating[2]) / (2 * np.pi)
# 将厂方校正图作用于当前扫描区域
corr = correction * padded_ind
# 对 256 取模确保灰度值在有效范围内
SLM_screen_Corrected = np.mod(padded_array, 256)
# 针对特定波长的相位缩放校正：215 可能是该 SLM 在当前波长下达到 2π 相位所需的灰度值
phase_to_screen = np.around(SLM_screen_Corrected * 215 / 255).astype('uint8')

# 将合成的局部扫描相位图更新至 SLM
slm.updateArray(phase_to_screen)

# --- 5. 全屏光栅显示 (系统对准用) ---
# 生成全屏的光栅相位，用于观察整体衍射效率或调整相机 ROI（感兴趣区域）
padded_array = 255 * generate_raster_matrix([1024, 1272]) / (2 * np.pi)
# 也可以切换为周期为 4 像素的斜光栅进行全屏显示
# padded_array = 255 * generate_tilted_grating([1024, 1272], period = 4) / (2 * np.pi)

# 同样进行取模和 SLM 特有的 8 位灰度级映射（215 缩放系数）
SLM_screen_Corrected = np.mod(padded_array, 256)
phase_to_screen = np.around(SLM_screen_Corrected * 215 / 255).astype('uint8')

# 将全屏光栅实时投射到 SLM 上
slm.updateArray(phase_to_screen)

# 初始化用于存储后续扫描数据的列表
list = []
x = []
y = []



#%%
# 这段代码最后获得了：SLM 上除中心块外的每一个 32×32小块，在 3 种不同的相移状态（0°、120°、240°）下，
# 与中心参考块干涉后所形成的 30 张（连拍）干涉光斑图像。

last_ir=(255*np.load('ir1.npy')/(2*np.pi))
rep=30
# 参考块（固定在中心）： 发出的光带有一级衍射角度，被透镜精准汇聚在相机的 [1200:1600, 3300:3700] 区域。
# 测试块（满屏幕跑）： 位置在跑，但发出的光和参考块的光是平行的。经过透镜后，它也丝毫不差地打在了相机的 [1200:1600, 3300:3700]。
for i in range(0, 1024//n):
    # 为当前i创建一个独立的数据存储结构
    current_i_lists = [[] for _ in range(s)]
    
    for j in range(1272 // m):
        # 排除特定块(i=16, j=19)
        if not (i == center[0] and j == center[1]):
            print(f"Processing i={i}, j={j}")
            
            for k in range(s):
                padded_ind=replace_block_padded(ind, i, j, np.ones((n,m)))
                padded_array = 255*replace_block_padded(array, i, j, grating[k])/(2*np.pi)
                # ir_corr=last_ir*padded_ind
                corr=correction*padded_ind
                SLM_screen_Corrected=np.mod(padded_array+corr,256)
                phase_to_screen=np.around(SLM_screen_Corrected*215/255).astype('uint8')

                # phase_to_screen = (215 * (padded_array / (2 * np.pi))).astype(np.uint8)
                slm.updateArray(phase_to_screen)
                
                time.sleep(0.75)
                list=[]
                for l in range (rep):
                    image=get_image()
                    # image=image[2000:2450,2350:2750]
                    # image=image[1350:1800,2450:2890]
                    image=image[1200:1600,3300:3700]   # 第一次进行图像裁剪，其实还能再小点，目前400*400

                    list.append(image.copy())
                    time.sleep(0.1)
                current_i_lists[k].append(list)
        if list[-1].max()<200:
            raise ValueError("diaole")    
    plt.imshow(current_i_lists[-1][18][-1])
    plt.show()
    filename = f"results5_i_{i}.pkl" 
    with open(filename, "wb") as f:
        pickle.dump(current_i_lists, f) 
    
    del current_i_lists



#%%
# 干涉实验的**“数据提纯与解包”**阶段
# .pkl文件：Pickle 可以把 Python 内存里的变量原封不动地“冻结”成二进制文件存到硬盘上
directory = "pkl"  # 替换为实际路径

# 遍历目录下所有以 results_i_ 开头的文件
all_data = {}
for filename in os.listdir(directory):
    if filename.startswith("results4_i_") and filename.endswith(".pkl"):
        # 提取 i 的值（例如从文件名 results_i_5.pkl 中提取 5）
        i = int(filename.split("_")[2].split(".")[0])
        
        # 加载文件内容
        with open(os.path.join(directory, filename), "rb") as f:
            all_data[i] = pickle.load(f)

for i in range(32):
    print(i)
    plt.imshow(all_data[i][-1][16][0])#第一个是i(行)，第二个是k(相位)，第三个是(j)列
    plt.show()

lists=[np.zeros((450,400)).astype(np.uint16)]*rep
for k in range(s):
    all_data[16][k].insert(19, lists)


#%%
X=[]
Y=[]
# 寻找“靶心”（质心法寻找光斑位置），这里发现光斑中心大约在[151,121]。
for j in range(32):
    for i in range (rep):
        x,y=calculate_centroid(all_data[j][-1][-1][i])
        X.append(x)
        Y.append(y)
print(sum(Y))

#%%
m0=[]
m1=[]
m2=[]
m3=[]
m4=[]
m5=[]
m6=[]
m7=[]
m8=[]
m9=[]
for i in range(1024//32):
    for j in range(1272 // 32):
        # if not (i == 16 and j == 19):
        n0=[]
        n1=[]
        n2=[]
        for k in range(rep):
            n0.append(all_data[i][0][j][k][151,121].astype(np.float32).sum())
            n1.append(all_data[i][1][j][k][151,121].astype(np.float32).sum())
            n2.append(all_data[i][2][j][k][151,121].astype(np.float32).sum())
        m0.append(filtered_mean(n0))
        m1.append(filtered_mean(n1))
        m2.append(filtered_mean(n2))
        # m3.append(all_data[i][3][j][142:145,169:172].astype(np.float32).sum())
        # m4.append(all_data[i][4][j][142:145,169:172].astype(np.float32).sum())
        # m5.append(all_data[i][5][j][142:145,169:172].astype(np.float32).sum())
        # m6.append(all_data[i][6][j][142:145,169:172].astype(np.float32).sum())
        # m7.append(all_data[i][7][j][142:145,169:172].astype(np.float32).sum())
        # m8.append(all_data[i][8][j][142:145,169:172].astype(np.float32).sum())
        # m9.append(all_data[i][9][j][142:145,169:172].astype(np.float32).sum())

        
#%%
# def calculate_phasor_12step(I):
#     """计算12步相移的相量 (I为长度12的数组)"""
#     # 背景项
#     A = np.mean(I)
    
#     # 实部计算 (利用对称性)
#     Re_p = ( I[0] 
#             + (np.sqrt(3)/2) * (I[1] + I[11])
#             + (1/2) * (I[2] + I[10])
#             - (1/2) * (I[4] + I[8])
#             - (np.sqrt(3)/2) * (I[5] + I[7])
#             - I[6] ) / 6
    
#     # 虚部计算 (利用反对称性)
#     Im_p = ( (1/2) * (I[1] - I[11])
#             + (np.sqrt(3)/2) * (I[2] - I[10])
#             + (I[3] - I[9])
#             + (np.sqrt(3)/2) * (I[4] - I[8])
#             + (1/2) * (I[5] - I[7]) ) / 6
            
#     return Re_p + 1j*Im_p, A

def calculate_phasor_3step(I):
    A = np.mean(I)

    Re_p = -(I[1]+I[2]-2*I[0])/3
    # 虚部计算 (利用反对称性)
    Im_p =(I[1]-I[2])/np.sqrt(3)
    return Re_p + 1j*Im_p, A

# 分行处理 数据太大了
# 之所以写这版代码，原因就在代码第一行的注释里：“分行处理 数据太大了”。
Amp=[]
phase=[]
for i in range(1024//32):
    # i=16
    directory = ""  # 替换为实际路径
    data=[]
    file_path = os.path.join(directory, f'results5_i_{i}.pkl')
    with open(file_path, 'rb') as f:  # 使用 with 确保安全打开文件
        data = pickle.load(f)
    plt.imshow(data[-1][16][0]) #第一个是i(行)，第二个是k(相位)，第三个是(j)列
    plt.show()

    # lists=[np.zeros((450,400)).astype(np.uint16)]*rep
    lists=[np.zeros((400, 400)).astype(np.uint16)]*rep

    if i==center[0]:
        for k in range(s):
            data[k].insert(center[1], lists)

    m=[[] for _ in range(s)]
    for j in range(1272 // 32):
        # if not (i == 16 and j == 19):
        n=[[] for _ in range(s)]
        for k in range(rep):
            for x in range(s):
                n[x].append(data[x][j][k][222,217].astype(np.float32).sum())

        for x in range(s):
            m[x].append(filtered_mean(n[x]))

    for j in range(1272 // 32):
        p,A=calculate_phasor_3step([row[j] for row in m])
        phi=-np.angle(p)
        amp=np.abs(p)
        phase.append(phi)
        Amp.append(amp)
phase=np.mod(phase,2*np.pi)
plt.imshow(np.abs(phase).reshape(32,39))
plt.colorbar()
plt.show()
plt.imshow(np.abs(Amp).reshape(32,39))
plt.colorbar()
plt.show()


#%%
# 这段代码其实是你在实验早期的**“标定工具”或“探路脚本”**。当你刚搭建好光路，不确定 SLM 响应是否线性、
# 不确定相机有没有非线性噪声时，你会特意扫个 6 步甚至 10 步，挑几个点拟合一下，
# 看看散点图是不是完美的余弦曲线（正如代码最后画图显示的那样）。
def cosine_func(x, A, phi, C):
    return A * np.cos(x + phi) + C
l=700
x_data = np.array([0, np.pi/6, 2*np.pi/6, 3*np.pi/6, 4*np.pi/6, 5*np.pi/6])
y_data = np.array([m0[l], m1[l], m2[l], m3[l], m4[l], m5[l]])  

# 初始猜测参数 [A, phi, C]
initial_guess = [6000, 0, 2000]

params, covariance = curve_fit(cosine_func, x_data, y_data, p0=initial_guess)
A_fit, phi_fit, C_fit = params

perr = np.sqrt(np.diag(covariance))
A_err, phi_err, C_err = perr[0], perr[1], perr[2]

print(f"拟合振幅: {A_fit:.3f} ± {A_err:.3f}")
print(f"x=0时的拟合相位: {phi_fit:.3f} ± {phi_err:.3f} 弧度")
print(f"常数项: {C_fit:.3f} ± {C_err:.3f}")
print(f"相位转换为角度: {np.degrees(phi_fit):.1f}° ± {np.degrees(phi_err):.1f}°")

# 可视化
x_fit = np.linspace(0, 5*np.pi/6, 100)
y_fit = cosine_func(x_fit, A_fit, phi_fit, C_fit)

plt.scatter(x_data, y_data, label='实际数据', color='red')
plt.plot(x_fit, y_fit, '--', label=f'拟合曲线: $A \cos(x+\\phi) + C$\n$A={A_fit:.2f}$, $\\phi={phi_fit:.2f}$, $C={C_fit:.2f}$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()



#%%
# 完整的十步位移
def estimate_initial_params(x, y):
    C_guess = np.median(y)
    y_centered = y - C_guess 
    A_guess = 0.5 * (np.max(y_centered) - np.min(y_centered))
    return A_guess, C_guess

def calculate_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def cosine_func(x, A, phi, C):
    return A * np.cos(x + phi) + C

def cos_fit(l,show=False):
    x_data = 2*np.array([0, np.pi/10, 2*np.pi/10, 3*np.pi/10, 4*np.pi/10, 5*np.pi/10,6*np.pi/10, 7*np.pi/10, 8*np.pi/10, 9*np.pi/10])
    y_data = np.array([m0[l], m1[l], m2[l], m3[l], m4[l], m5[l], m6[l], m7[l], m8[l], m9[l]]) 
    A_guess,C_guess=estimate_initial_params(x_data,y_data)
    initial_guess = [A_guess, np.pi/2, C_guess]  # [A_guess, k_guess, phi_guess, C_guess]
    # initial_guess=[A_guess,0.78,np.pi/2,C_guess]
    threshold = 0.95
    # lower_bounds = [0,   0, 0, 0]  # [A_min, k_min, phi_min, C_min]
    # upper_bounds = [np.inf, np.inf,  2*np.pi,  np.inf] # [A_max, k_max, phi_max, C_max]
    try:
        params, covariance = curve_fit(
            cosine_func, x_data, y_data,
            p0=initial_guess,
            maxfev=1000
            # bounds=(lower_bounds, upper_bounds)
        )
        
        R=calculate_r_squared(y_data,cosine_func(x_data, *params))
        k_fit=1
        k_err=1
        if R < threshold:
            # print("残差过大，拟合结果无效，输出0")
            flag=False
            return flag,R,0,0,0,0
        else:
            # print("拟合成功！")
            flag=True
            A_fit, phi_fit, C_fit = params
            perr = np.sqrt(np.diag(covariance))
            A_err, phi_err, C_err = perr[0], perr[1], perr[2]
    except RuntimeError:
        flag=False
        # print("迭代次数超限，拟合失败，输出0")
        return flag,0,0,0,0,0
    except Exception as e:
        flag=False
        print(f"未知错误: {e}，输出0")
        return flag,0,0,0,0,0

    if (flag and show):

        print(f"拟合振幅 (A): {A_fit:.3f} ± {A_err:.3f}")
        print(f"频率系数 (k): {k_fit:.3f} ± {k_err:.3f}")
        print(f"相位 (phi): {phi_fit:.3f} ± {phi_err:.3f} 弧度")
        print(f"常数项 (C): {C_fit:.3f} ± {C_err:.3f}")
        print(f"相位转换为角度: {np.degrees(phi_fit):.1f}° ± {np.degrees(phi_err):.1f}°")
        
        x_fit = np.linspace(0, 2*np.pi, 200)
        y_fit = cosine_func(x_fit, A_fit, phi_fit, C_fit)
        
        plt.scatter(x_data, y_data, label='数据', color='red', alpha=0.5)
        plt.plot(x_fit, y_fit, '--', label=f'拟合曲线: $A \cos(kx + \\phi) + C$\n$A={A_fit:.2f}$, $k={k_fit:.2f}$, $\\phi={phi_fit:.2f}$, $C={C_fit:.2f}$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()
    return flag,A_fit,phi_fit,C_fit,k_fit,R

#%%
mask=[]
Amp=[]
phi=[]
c=[]
k=[]
for i in range((1024//32)*(1272//32)):
    flag,amp,phase,C_fit,k_fit,R=cos_fit(i)
    Amp.append(amp)
    phi.append(phase)
    c.append(C_fit)
    k.append(k_fit)
    mask.append(flag)
mask=np.array(mask)   
Amp=np.array(Amp) 
phi=np.array(phi) 
c=np.array(c) 
k=np.array(k)

Amp_ind=np.where(Amp<0)
k_ind=np.where(k<0)
c_ind=np.where(c<0)
print(c_ind)

for i in k_ind:
    phi[i]=-phi[i]

for i in Amp_ind:
    phi[i]=phi[i]+np.pi
phi=np.mod(phi,2*np.pi)
plt.imshow(np.abs(phi).reshape(32,39))
plt.colorbar()
plt.show()
plt.imshow(np.abs(Amp).reshape(32,39))
plt.colorbar()
plt.show()



#%%
phi=[]
P=[]
for i in range((1024//32)*(1272//32)):
    p=-1/3*(m1[i]+m2[i]-2*m0[i])+1j/np.sqrt(3)*(m1[i]-m2[i])
    P.append(p)
    phi.append(np.angle(p))
    # phi.append()

#%%
#unwrap
phase_reshaped=np.array(phase).reshape(32,39)
# def has_nonzero_neighbor(patch):
#     """
#     检查3x3邻域内是否存在非零值（中心为当前像素）
#     """
#     center = patch[4]  # 中心像素值
#     neighbors = np.delete(patch, 4)  # 移除中心值
#     return 1 if (center == 0) and (np.any(neighbors != 0)) else 0

# # 应用滑动窗口检测
# neighbor_mask = generic_filter(
#     mask.reshape(32,39), 
#     has_nonzero_neighbor, 
#     size=3, 
#     mode='constant'
# )
# neighbor_mask = neighbor_mask.astype(bool)

# # 步骤2：修复无效区域（双调和插值）
# phase_repaired = inpaint.inpaint_biharmonic(
#     phi, 
#     mask=neighbor_mask  # 需要修复的区域
# )

unwrapped = unwrap_phase(phase_reshaped)
height, width = unwrapped.shape[:2]
img_resized = cv2.resize(
    unwrapped,
    (width * 32, height * 32),
    interpolation=cv2.INTER_LANCZOS4  # 相对高质量的插值
)
img_resizedb = cv2.resize(
    unwrapped,
    (width * 4, height * 4),
    interpolation=cv2.INTER_LANCZOS4  # 相对高质量的插值
)
height2, width2=img_resizedb.shape[:2]

img_resized2 = cv2.resize(
    img_resizedb,
    (width2 * 8, height2 * 8),
    interpolation=cv2.INTER_LANCZOS4  # 相对高质量的插值
)
# %%
# 找到掩膜的边界坐标
coords = np.argwhere(neighbor_mask)
y_min, x_min = coords.min(axis=0)
y_max, x_max = coords.max(axis=0)

# # 提取目标区域子矩阵
A_sub = unwrapped[y_min:y_max+1, x_min:x_max+1]
mask_sub = neighbor_mask[y_min:y_max+1, x_min:x_max+1]
# %%
# 定义放大因子
SCALE_FACTOR = 32

# 双三次插值放大目标区域
A_sub_highres = cv2.resize(
    A_sub, 
    (SCALE_FACTOR*A_sub.shape[1], SCALE_FACTOR*A_sub.shape[0]),
    interpolation=cv2.INTER_CUBIC
)

# 生成高分辨率掩膜 (使用线性插值+阈值)
mask_sub_highres = cv2.resize(
    mask_sub.astype(np.float32), 
    (SCALE_FACTOR*A_sub.shape[1], SCALE_FACTOR*A_sub.shape[0]),
    interpolation=cv2.INTER_LINEAR
)
mask_sub_highres = (mask_sub_highres > 0.5).astype(bool)
# %%
# 全图最近邻放大
A_lowres = cv2.resize(
    unwrapped, 
    (SCALE_FACTOR*unwrapped.shape[1], SCALE_FACTOR*unwrapped.shape[0]),
    interpolation=cv2.INTER_NEAREST
)
# %%
# 计算目标区域在放大矩阵中的位置
y1 = y_min * SCALE_FACTOR
y2 = (y_max + 1) * SCALE_FACTOR
x1 = x_min * SCALE_FACTOR
x2 = (x_max + 1) * SCALE_FACTOR

# 确保尺寸匹配
assert A_sub_highres.shape[0] == (y2 - y1)
assert A_sub_highres.shape[1] == (x2 - x1)

# 创建替换区域副本
replace_area = A_lowres[y1:y2, x1:x2].copy()

# 使用高分辨率数据覆盖目标区域
replace_area[mask_sub_highres] = A_sub_highres[mask_sub_highres]

# 更新最终矩阵
A_final = A_lowres.copy()
A_final[y1:y2, x1:x2] = replace_area
plt.figure(figsize=(18,6))

# 原始数据与掩膜
plt.subplot(131)
plt.imshow(unwrapped, cmap='viridis')
plt.title('原始矩阵')
plt.colorbar()

# 插值区域对比
plt.subplot(132)
plt.imshow(A_lowres, cmap='viridis')  # 中心区域低分辨率
plt.title('普通放大区域')
plt.colorbar()

plt.subplot(133)
plt.imshow(A_final, cmap='viridis')  # 中心区域高分辨率
plt.title('局部高密度插值区域')
plt.colorbar()
plt.show()







#%%
def expand_matrix(matrix, n, m):
    # 先按行重复 n 次，再按列重复 m 次
    expanded = np.repeat(matrix, n, axis=0)
    expanded = np.repeat(expanded, m, axis=1)
    return expanded
phi=phi.reshape(32,39)
# 示例
n, m = 32, 32  # 每个像素扩展为 n行×m列 的块
slm_phase = expand_matrix(phi, n, m)
slm_phase=np.pad(slm_phase,((0,0),(12,12)))
#%%
p=[]
for i in range(x):
    s=-1/3*(m2[i]+m3[i]-2*m1[i])+1j*((m2[i]-m3[i])/np.sqrt(3))
    p.append(s)
angle=[]
for i in range(x):
    angle.append(p[i].angel)
#%%
lists = [[] for _ in range(s)]
for i in range(1024//32):
    print(i)
    for j in range(1272//32):
        if i!=16 or j!=19:
            for k in range(s):
                padded_array = replace_block_padded(array, i, j, grating[k])
                phase_to_screen=(215*(padded_array/(2*np.pi))).astype(np.uint8)
                slm.updateArray(phase_to_screen)
                time.sleep(0.2)
                image = IDS_Camera.GetImage()
                lists[k].append(image.copy())


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
    x = torch.arange(rows, dtype=torch.float64) - rows / 2
    y = torch.arange(cols, dtype=torch.float64) - cols / 2
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

#%%
slm_phase=torch.from_numpy(generate_raster_matrix([1024,1272],64,64))-np.pi
L, Lp = laser_gaussian(dim=[1024,1272], r0=(5, 5), sigmax=4*1000/12.5, sigmay=4*1000/12.5, A=1.0, save_param=True)

I_L_tot = torch.sum(torch.pow(L, 2.))                           #
L = L * torch.pow(1 / I_L_tot, 0.5)                      #
I_L_tot = torch.sum(torch.pow(L, 2.))

slm = torch.multiply(L,(torch.exp(1j*(slm_phase))))
slm_shift = torch.fft.fftshift(slm)#
fft_field = torch.fft.fft2(slm_shift)
fft_field_shift = torch.fft.fftshift(fft_field).cpu().clone()

fft_phase = fft_field_shift.angle()
fft_amp = fft_field_shift.abs()
fft_intensity = fft_amp**2
# %%
def centroid(data):
    total = np.sum(data)
    x_c = np.sum(data * x) / total
    y_c = np.sum(data * y) / total
    return x_c, y_c

# 高斯拟合
def gaussian_2d(xy, A, x0, y0, sigma, B):
    x, y = xy
    return A * np.exp(-((x - x0)**2 + (y - y0)**2)/(2 * sigma**2)) + B

x, y = np.meshgrid(np.arange(300), np.arange(300))
data= all_data[30][0][-1]
p0 = [100, 100, 100, 20, 0]  # 初始猜测
params, _ = curve_fit(gaussian_2d, (x.ravel(), y.ravel()), data.ravel(), p0=p0)
x0_fit, y0_fit = params[1], params[2]


# 输出结果
print(f"质心法中心: ({centroid(data)[0]:.2f}, {centroid(data)[1]:.2f})")
print(f"高斯拟合中心: ({x0_fit:.2f}, {y0_fit:.2f})")
# %%
def xie_phase(n, r0, save_param=False):

    # initialization
    x = np.array(list(range(n[0])))*1.
    y = np.array(list(range(n[1])))*1.   
    X, Y = np.meshgrid(x, y)
    z = np.zeros((n[0],n[1]))

    # target definition
    z = np.mod(12.5/100*((X-r0[0])*np.sqrt(3)+(Y-r0[0]))/2, 2*np.pi)

    return torch.from_numpy(z)
# %%
plt.imshow(xie_phase([1272,1024],[636,512]))
plt.colorbar()
# %%
# path='8134.npy'
# phase=torch.angle(torch.exp(1j*xie_phase([1272,1024],[636,512]))+torch.exp(1j*torch.from_numpy(np.load(path))))
# phase=torch.remainder(phase,2*torch.pi)
# phase=torch.angle(torch.exp(1j*xie_phase([1272,1024],[636,512]))+torch.exp(1j*phase))
# phase=torch.remainder(phase,2*torch.pi)

# #%%
# # tophatscreen=(255*np.load(path)/(2*np.pi))
# tophatscreen=(255*np.array(phase)/(2*np.pi))

tophatscreen=255*np.array(np.load('1.npy'))/(2*np.pi)
tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
# slmx1 = slmpy.SLMdisplay(monitor=0,isImageLock = True)
slm.updateArray(tophat_screen_Corrected)
# %%
A_sub_highres = cv2.resize(
    amp, 
    (32*amp.shape[1], 32*amp.shape[0]),
    interpolation=cv2.INTER_CUBIC
)
#%%
pad_width = ((0, 0), (12, 12))  # (高度前后, 宽度左右)
padded_arr = np.pad(
    img_resized,
    pad_width=pad_width, 
    mode='constant', 
    constant_values=0
)
#%%
set_param(exposure=200)
# %%
slm = slmpy.SLMdisplay(monitor=0,isImageLock = True)

#%%
# path='813_eff4.npy'
# path='tar_813_rand2.npy'
path='test5.npy'
# path='813_3.45c.npy'
phase_end=np.load('ir1.npy')
phases=np.mod(np.load(path)-padded_arr,2*np.pi)
tophatscreen=(255*phases/(2*np.pi))

# tophatscreen=(255*phase/(2*np.pi))
# tophatscreen=255*np.array(cg1.slm_phase_end)/(2*np.pi)
tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
# tophat_screen_Corrected=(215*tophatscreen/255).astype('uint8')
slmx2.updateArray(tophat_screen_Corrected)
# %%
















#下面是新年快乐
#%%
time.sleep(6)
img1=[]
set_param(exposure=1050)
for i in range(11):
    tophatscreen=255*np.array(np.load(f'phase//phase1//move_to_bighappy{i}.npy'))/(2*np.pi)
    tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
    # slmx1 = slmpy.SLMdisplay(monitor=0,isImageLock = True)
    slm.updateArray(tophat_screen_Corrected)
    time.sleep(1)
    image=get_image()
    image=image[1550:2500,500:2200]
    images = rotate(image,angle=11.3,reshape=False)
    images=np.flip(images, axis=1)
    img1.append(images.copy())
#%%
img2=[]
set_param(exposure=400)
time.sleep(2)
for i in range(5):
    tophatscreen=255*np.array(np.load(f'phase//happy_middle{i}.npy'))/(2*np.pi)
    tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
    # slmx1 = slmpy.SLMdisplay(monitor=0,isImageLock = True)
    slm.updateArray(tophat_screen_Corrected)
    time.sleep(1)
    image=get_image()
    image=image[1550:2500,500:2200]
    images = rotate(image,angle=11.3,reshape=False)
    images=np.flip(images, axis=1)
    img2.append(images.copy())
for i in range(3,14):
    tophatscreen=255*np.array(np.load(f'phase//happy_flame_{i}.npy'))/(2*np.pi)
    tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
    # slmx1 = slmpy.SLMdisplay(monitor=0,isImageLock = True)
    slm.updateArray(tophat_screen_Corrected)
    time.sleep(1)
    image=get_image()
    image=image[1550:2500,500:2200]
    images = rotate(image,angle=11.3,reshape=False)
    images=np.flip(images, axis=1)
    img2.append(images.copy())

#%%
img3=[]
set_param(exposure=1250)
time.sleep(2)
for i in range(11):
    tophatscreen=255*np.array(np.load(f'phase//move_to_biggong{i}.npy'))/(2*np.pi)
    tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
    # slmx1 = slmpy.SLMdisplay(monitor=0,isImageLock = True)
    slm.updateArray(tophat_screen_Corrected)
    time.sleep(1)
    image=get_image()
    image=image[1550:2500,500:2200]
    images = rotate(image,angle=11.3,reshape=False)
    images=np.flip(images, axis=1)
    img3.append(images.copy())

#%%
img4=[]
set_param(exposure=450)
time.sleep(2)
for i in range(5):
    tophatscreen=255*np.array(np.load(f'phase//gong_middle{i}.npy'))/(2*np.pi)
    tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
    # slmx1 = slmpy.SLMdisplay(monitor=0,isImageLock = True)
    slm.updateArray(tophat_screen_Corrected)
    time.sleep(1)
    image=get_image()
    image=image[1550:2500,500:2200]
    images = rotate(image,angle=11.3,reshape=False)
    images=np.flip(images, axis=1)
    img4.append(images.copy())

for i in range(3,14):
    tophatscreen=255*np.array(np.load(f'phase//gong_flame_{i}.npy'))/(2*np.pi)
    tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
    # slmx1 = slmpy.SLMdisplay(monitor=0,isImageLock = True)
    slm.updateArray(tophat_screen_Corrected)
    time.sleep(1)
    image=get_image()
    image=image[1550:2500,500:2200]
    images = rotate(image,angle=11.3,reshape=False)
    images=np.flip(images, axis=1)
    img4.append(images.copy())
# %%


time.sleep(6)
img1b=[]
set_param(exposure=500)
for i in range(11):
    tophatscreen=255*np.array(np.load(f'phase//move_to_newyear{i}b.npy'))/(2*np.pi)
    tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
    # slmx1 = slmpy.SLMdisplay(monitor=0,isImageLock = True)
    slm.updateArray(tophat_screen_Corrected)
    time.sleep(1)
    image=get_image()
    image=image[550:2600,600:2600]
    images = rotate(image,angle=10.5,reshape=False)
    images=np.flip(images, axis=1)
    img1b.append(images.copy())
#%%
img2b=[]
set_param(exposure=190)
time.sleep(2)
for i in range(5):
    tophatscreen=255*np.array(np.load(f'phase//happy_middle{i}b.npy'))/(2*np.pi)
    tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
    # slmx1 = slmpy.SLMdisplay(monitor=0,isImageLock = True)
    slm.updateArray(tophat_screen_Corrected)
    time.sleep(1)
    image=get_image()
    image=image[550:2600,600:2600]
    images = rotate(image,angle=10.5,reshape=False)
    images=np.flip(images, axis=1)
    img2b.append(images.copy())
for i in range(3,14):
    tophatscreen=255*np.array(np.load(f'phase//happy_flame_{i}b.npy'))/(2*np.pi)
    tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
    # slmx1 = slmpy.SLMdisplay(monitor=0,isImageLock = True)
    slm.updateArray(tophat_screen_Corrected)
    time.sleep(1)
    image=get_image()
    image=image[550:2600,600:2600]
    images = rotate(image,angle=10.5,reshape=False)
    images=np.flip(images, axis=1)
    img2b.append(images.copy())

#%%
img3b=[]
set_param(exposure=600)
time.sleep(2)
for i in range(11):
    tophatscreen=255*np.array(np.load(f'phase//move_to_biggong{i}b.npy'))/(2*np.pi)
    tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
    # slmx1 = slmpy.SLMdisplay(monitor=0,isImageLock = True)
    slm.updateArray(tophat_screen_Corrected)
    time.sleep(1)
    image=get_image()
    image=image[550:2600,600:2600]
    images = rotate(image,angle=10.5,reshape=False)
    images=np.flip(images, axis=1)
    img3b.append(images.copy())

#%%
img4b=[]
set_param(exposure=220)
time.sleep(2)
for i in range(5):
    tophatscreen=255*np.array(np.load(f'phase//gong_middle{i}b.npy'))/(2*np.pi)
    tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
    # slmx1 = slmpy.SLMdisplay(monitor=0,isImageLock = True)
    slm.updateArray(tophat_screen_Corrected)
    time.sleep(1)
    image=get_image()
    image=image[550:2600,600:2600]
    images = rotate(image,angle=10.5,reshape=False)
    images=np.flip(images, axis=1)
    img4b.append(images.copy())

for i in range(3,14):
    tophatscreen=255*np.array(np.load(f'phase//gong_flame_{i}b.npy'))/(2*np.pi)
    tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
    # slmx1 = slmpy.SLMdisplay(monitor=0,isImageLock = True)
    slm.updateArray(tophat_screen_Corrected)
    time.sleep(1)
    image=get_image()
    image=image[550:2600,600:2600]
    images = rotate(image,angle=10.5,reshape=False)
    images=np.flip(images, axis=1)
    img4b.append(images.copy())
# %%
np.save('xinina_list1bc.npy',img1b)
np.save('xinina_list2bc.npy',img2b)
np.save('xinina_list3bc.npy',img3b)
#%%
np.save('xinina_list4bc.npy',img4b)
# %%
