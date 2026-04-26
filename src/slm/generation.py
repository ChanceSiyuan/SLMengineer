#%%
import numpy as np
import scipy as sp
from scipy import fft
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import os
import time
import json
from pathlib import Path
from slm.aberration import Zernike

# import slmpy


class SLM_class():  #用于生成输入输出振幅分布
    """Unified SLM target and phase generator.

    Holds hardware configuration (from ``hamamatsu_test_config.json``),
    generates focal-plane targets, and converts optimised phase maps into
    SLM-sized screens.  All target-producing methods return complex-valued
    arrays (``np.complex128``) so a single target can feed both WGS and
    CGM.  Spot-array targets have real-positive amplitude and zero phase.

    Target-method compatibility:

        Method                          | WGS | CGM
        --------------------------------|-----|-----
        target_generate("Rec" | "Tri")  | yes | yes   discrete spots, zero phase
        RecLattice / triangular_lattice | yes | yes
        translate/rotate/modify_target  | yes | yes   preserves input dtype
        top_hat_target                  | no  | yes   continuous flat region
        gaussian_line_target            | no  | yes   phase gradient is load-bearing
        light_sheet_target              | no  | yes
        lg_mode_target                  | no  | yes   vortex phase
        square_lattice_vortex_target    | no  | yes
        ring_lattice_vortex_target      | no  | yes
        graphene_lattice_target         | no  | yes   alternating 0/π phase

    "yes" under WGS means ``WGS_phase_generate`` gives a sensible result
    (it uses only ``|target|``).  "no" means the target's phase is
    essential to the pattern and WGS would collapse it to a meaningless
    intensity-only map.
    """

    def __init__(self):
        module_dir = Path(__file__).resolve().parent
        repo_root = module_dir.parent.parent
        config_candidates = [
            Path.cwd() / "hamamatsu_test_config.json",
            repo_root / "hamamatsu_test_config.json",
            module_dir / "hamamatsu_test_config.json",
        ]
        config_path = next((p for p in config_candidates if p.exists()), None)
        if config_path is None:
            raise FileNotFoundError(
                "Could not find 'hamamatsu_test_config.json'. "
                "Checked current working directory, repo root, and src/slm."
            )

        with open(config_path, 'r') as file:
            data = json.load(file)
        self.pixelpitch = data['pixelpitch']  # SLM像素尺寸，单位um
        self.SLMRes = data['SLMRes']  # SLM屏幕分辨率

        self.arraySizeBit = data['arraySizeBit']  # 计算用图片尺寸深度，一般用11或12
        self.Loop = data['Loop']  # 循环次数
        self.threshold = data['threshold']  # 相位固定阈值

        self.beamwaist = data['beamwaist']  # 入射高斯光束腰，单位um
        self.focallength = data['focallength']  # 物镜焦距，单位um
        self.magnification = data['magnification']  # SLM到物镜间的成像系统的放大率
        self.wavelength = data['wavelength']  # 光波长，单位um
        self.mask = data['mask']  # 是否有光阑
        self.maskradius = data['maskradius']  # 光阑半径，单位um

        self.distance = data['distance']  # 阵列中最近点到原点的真实物理距离，单位um，注意必须为正值，否则相机反馈时顺序会乱
        self.spacing = data['spacing']  # 阵列间隔，单位um
        self.arraysize = data['arraysize']  # 阵列数量

        self.translate = data['translate']  # 将SLM阵列移到屏幕中央，注意此时输入的目标图应该是未旋转过的
        self.rotate = data['rotate']  # 旋转SLM阵列，来匹配AOD阵列。请注意！！！旋转后可能会使相机反馈输入顺序错乱，到时要注意好顺序！！！
        self.angle = data['angle']  # 旋转角度，单位：度
        self.modify = data['modify']  # 根据SLM的衍射效率作对目标图作修正，如果阵列体系较大，建议使用
        
        self.AddZernike = data['AddZernike']  # 是否加上Zernike修正
        self.ind_Zernike_list = data['ind_Zernike_list']  # 需要添加的Zernike项，注意项数从0到27，对应Zernike的径向n从0到6。注意该图显示是1-28，但实际函数调用应该是0-27
        self.percent_list = data['percent_list']  # percent表示对应Zernike项的强度，范围-1到1。物理上，它对应着对应像差的rms值
        self.zernike_aperture_radius = data['zernike_aperture_radius']  # 要添加的Zernike圆半径，通常与光阑半径一致，单位um
        self.isZernikePhaseContinous = data['isZernikePhaseContinous']  # 是否需要将圆形的Zernike相位连续的扩展到方形来匹配SLM尺寸，通常不需要 


    # def  open(self):
    #     self.slm = slmpy.SLMdisplay(monitor=1,isImageLock = True)

    
    # def close(self):
    #     self.slm.close()


    # def updateArray(self, SLM_screen_Corrected):
    #     self.slm.updateArray(SLM_screen_Corrected)


    def image_init(self, initGaussianPhase_user_defined = None, initGaussianPhase_save = False, Plot = True, beam_center_um = (0.0, 0.0)):
        self.arraySizeBitx = self.arraySizeBit[0]
        self.arraySizeBity = self.arraySizeBit[1]
        self.ImgResX = 2 ** (int(self.arraySizeBitx))
        self.ImgResY = 2 ** (int(self.arraySizeBity))
        self.Xps, self.Yps = np.meshgrid(np.linspace(0, self.ImgResX, self.ImgResX), np.linspace(0, self.ImgResY, self.ImgResY))
        self.Focalpitchx = self.wavelength*self.focallength/self.ImgResX/(self.pixelpitch*self.magnification)#
        self.Focalpitchy = self.wavelength*self.focallength/self.ImgResY/(self.pixelpitch*self.magnification)
        # beam_center_um = (dx, dy): incident-beam center offset from the
        # SLM compute-grid center, in um.  Used to model off-center
        # illumination so CGM optimizes against the real input amplitude.
        self.beam_center_um = tuple(beam_center_um)
        dx_um, dy_um = self.beam_center_um
        X = self.Xps*self.pixelpitch-self.ImgResX/2*self.pixelpitch - dx_um
        Y = self.Yps*self.pixelpitch-self.ImgResY/2*self.pixelpitch - dy_um

        initGaussianAmp = np.sqrt(2 / np.pi) / self.beamwaist * np.exp(-(X ** 2 + Y ** 2) / self.beamwaist ** 2)

        if initGaussianPhase_user_defined is not None:
            self.initGaussianPhase = initGaussianPhase_user_defined
        else:
            self.initGaussianPhase = np.random.rand(self.ImgResX, self.ImgResY) * 2 * np.pi - np.pi

        if self.mask:
            maskAmp = (X**2+Y**2 <= self.maskradius**2)*1
            initGaussianAmp = np.multiply(initGaussianAmp, maskAmp)
            self.initGaussianPhase = np.multiply(self.initGaussianPhase, maskAmp)


        initIntensity = np.square(initGaussianAmp)
        self.initGaussianAmp = initGaussianAmp / np.sqrt(np.sum(initIntensity))

        if initGaussianPhase_save:
            np.save(self.path1+'/'+self.Time+'_'+str(self.arraysize[0])+'x'+str(self.arraysize[1])+'_initGaussianPhase.npy',self.initGaussianPhase)

        print('Focal pixelpitch:',[self.Focalpitchx,self.Focalpitchy])
        print(f'The input gaussian light beam waist: {self.beamwaist}um')
        if self.mask:
            print(f'Please note: you have used mask with radius {self.maskradius}um')
        if initGaussianPhase_user_defined is not None:
            print('You have specified the init gaussian phase.')
        else:
            print('The init gaussian phase is random generated.')

        if Plot:
            plt.imshow(self.initGaussianAmp)
            plt.colorbar()
            plt.title("Init gaussian amplitude")


    # def target_generate(self, Lattice_type, distance = None, spacing = None, arraysize = None, translate = None):
    #     if Lattice_type == 'Rec':
    #         fpx,fpy,targetAmp,location=self.SLMimage.initFocalImage_RecLattice(self.distance, self.spacing, self.arraysize, Plot=False)  # 获取直方形阵列的目标图
    #     else:
    #         raise('Lattice type not found!')
        
    #     if self.translate:
    #         targetAmp,location = self.SLMimage.translate_targetAmp(targetAmp,location)
    #         print('You have translated the target to the image center.')
    #     if self.rotate:
    #         targetAmp,location = self.SLMimage.rotate_targetAmp(targetAmp,self.angle,location)
    #         print(f'You have rotated the target by angle {self.angle}.')
    #     if self.modify:
    #         targetAmp = self.SLMimage.modify_targetAmp(targetAmp)
    #         print('You have modified the target to compensate the diffraction efficiency.')
        
    #     position = np.where(targetAmp>0)

    #     return targetAmp, location, position
    
    def target_generate(self, Lattice_type, distance = None, spacing = None, arraysize = None, translate = None, rotate = None, angle = None, modify = None, Plot = False,exception=None):
        distance = distance or self.distance
        spacing = spacing or self.spacing
        arraysize = arraysize or self.arraysize
        translate = translate or self.translate
        rotate = rotate or self.rotate
        angle = angle or self.angle
        modify = modify or self.modify
        
        
        if Lattice_type == 'Rec':

            if translate:
                targetAmp = self.RecLattice(distance, spacing, arraysize, Plot=False,exception=exception)
                targetAmp = self.translate_targetAmp(targetAmp,Plot=Plot)
                print('You have translated the target to the image center.')
            else:
                targetAmp = self.RecLattice(distance, spacing, arraysize, Plot=Plot,exception=exception)
                print(f'You have generated the target with {distance}um distance relative to the center.')
            if rotate:
                targetAmp = self.rotate_targetAmp(targetAmp,angle,Plot=Plot)
                print(f'You have rotated the target by angle {angle}.')
            if modify:
                targetAmp = self.modify_targetAmp(targetAmp)
                print('You have modified the target to compensate the diffraction efficiency.')

        elif Lattice_type == 'Tri':

            if translate:
                targetAmp = self.triangular_lattice(distance, spacing, arraysize, Plot=False)
                targetAmp = self.translate_targetAmp(targetAmp,Plot=Plot)
                print('You have translated the target to the image center.')
            else:
                targetAmp = self.triangular_lattice(distance, spacing, arraysize, Plot=Plot)
                print(f'You have generated the target with {distance}um distance relative to the center.')
            if rotate:
                targetAmp = self.rotate_targetAmp(targetAmp,angle,Plot=Plot)
                print(f'You have rotated the target by angle {angle}.')
            if modify:
                targetAmp = self.modify_targetAmp(targetAmp)
                print('You have modified the target to compensate the diffraction efficiency.')
        
        else:
            raise('Lattice type not found!')

        return targetAmp
    

    def RecLattice(self, distance, spacing, arraysize, Plot=False,exception=None):
        targetAmp = np.zeros((int(self.ImgResY), int(self.ImgResX)), dtype=np.complex128)
        dm = round(distance[0]/self.Focalpitchx)
        dn = round(distance[1]/self.Focalpitchy)
        mcenter = self.ImgResX/2
        ncenter = self.ImgResY/2
        m = mcenter + dm
        n = ncenter + dn

        totalsitesnum = arraysize[0]*arraysize[1]
        intensityPerSite = 1/totalsitesnum

        arraysizex = arraysize[0]
        arraysizey = arraysize[1]

        spacingx = round(spacing[0]/self.Focalpitchx)
        spacingy = round(spacing[1]/self.Focalpitchy)

        print('focal pixelpitch:',[self.Focalpitchx,self.Focalpitchy])
        print('array spacing:',spacing)
        print('spacing pixels:',[spacingx,spacingy])
        print('arraysize:',arraysize)
        startRow = int(n-(arraysizey-1)*spacingy/2)
        endRow = int(startRow+(arraysizey-1)*spacingy)
        startCol = int(m-(arraysizex-1)*spacingx/2)
        endCol = int(startCol+(arraysizex-1)*spacingx)

        if startRow < 0 or startCol < 0:
            raise ValueError("Sorry, too big an array, consider shrinking the spacing or the size!")

        targetAmp[startRow:endRow+spacingy:spacingy,:][:, startCol:endCol+spacingx:spacingx]=intensityPerSite**0.5
        if exception:
            for i in exception:
                targetAmp[startRow+spacingy*i[0], startCol+spacingx*i[1]]=0
    

        if Plot:
            self.plot_target(targetAmp)

        return targetAmp
    


    def triangular_lattice(self, distance, spacing, arraysize, Plot=False):
        targetAmp = np.zeros((int(self.ImgResY), int(self.ImgResX)), dtype=np.complex128)

        dm = np.round(distance/np.sqrt(2)/self.Focalpitchx)
        dn = np.round(distance/np.sqrt(2)/self.Focalpitchy)
        mcenter = self.ImgResX/2
        ncenter = self.ImgResY/2
        m = mcenter-dm
        n = ncenter - dn

        totalsitesnum = arraysize[0]*arraysize[1]
        intensityPerSite = 1/totalsitesnum

        arraysizex = arraysize[0]
        arraysizey = arraysize[1]

        spacingx = np.round(spacing[0]/self.Focalpitchx)
        spacingy = np.round((spacing[1]/self.Focalpitchy)*np.sqrt(3)*0.5)

        print('focal pixelpitch:',[self.Focalpitchx,self.Focalpitchy])
        print('array spacing:',spacing)
        print('spacing pixels:',[spacingx,spacingy])
        print('arraysize:',arraysize)

        startRow = int(n-(arraysizey-1)*spacingy)
        endRow = int(n)
        startCol = int(m-(arraysizex-1)*spacingx)
        endCol = int(m)

        try:
            if startRow < 0 or startCol < 0:
                raise ValueError("Sorry, too big an array, consider shrinking the spacing or the size!")
        except ValueError as ve:
            print(ve)

        # for i in range(arraysizey):
        #     if i % 2 == 0:
        #         targetAmp[startRow:endRow+1:spacingy, startCol+i*spacingx:endCol+1:spacingx] = intensityPerSite**0.5
        #     else:
        #         targetAmp[startRow+spacingy//2:endRow+1+spacingy//2:spacingy, startCol+i*spacingx:endCol+1:spacingx] = intensityPerSite**0.5
        for i in range(arraysizex):
            for j in range(arraysizey):
                offset_x = i * spacingx
                offset_y = j * spacingy

                if j % 2 == 1:
                    offset_x += spacingx // 2

                targetAmp[startRow + offset_x, startCol + offset_y] = intensityPerSite**0.5
                
        if Plot:
            plt.imshow(targetAmp, cmap='Greys')
            plt.title('Triangular Lattice')
            plt.show()

        return targetAmp


    def translate_targetAmp(self, targetAmp, Plot=False):
        '''
        This function translate the target amplitude such that it is centered on the screen.

        Input target Amp should be with no zero order offset and no rotation
        '''

        targetAmp_new = np.zeros_like(targetAmp)

        totalRow = np.size(targetAmp, axis=0)
        totalCol = np.size(targetAmp, axis=1)


        point = np.where(targetAmp)
        startRow = point[0][0]
        endRow = point[0][-1]
        startCol = point[1][0]
        endCol = point[1][-1]
        centerRow = (startRow+endRow)/2
        centerCol = (startCol+endCol)/2

        deltaRow = int(totalRow/2 - centerRow)
        deltaCol = int(totalCol/2 - centerCol)

        targetAmp_new[(np.where(targetAmp)[0]+deltaRow,np.where(targetAmp)[1]+deltaCol)] = targetAmp[np.where(targetAmp)]

        if Plot:
            self.plot_target(targetAmp_new)

        return targetAmp_new
    

    def rotate_targetAmp(self, targetAmp, angle, Plot=False):
        '''
        This function rotates the target Amp pattern around the origin. 
        
        The goal is to match SLM traps with AOD sorting traps. 
        
        targetAmp is the input target foci pattern,angle is an input value in degree.
        '''

        targetAmp_new = np.zeros_like(targetAmp)

        totalRow = np.size(targetAmp, axis=0)
        totalCol = np.size(targetAmp, axis=1)

        point = np.where(targetAmp)
        pointRow = point[0]
        pointCol = point[1]

        indice1 = np.where((np.argwhere(targetAmp)==[pointRow[0],pointCol[0]]).all(axis=1))[0][0]
        indice2 = np.where((np.argwhere(targetAmp)==[pointRow[0],pointCol[-1]]).all(axis=1))[0][0]
        indice3 = np.where((np.argwhere(targetAmp)==[pointRow[-1],pointCol[-1]]).all(axis=1))[0][0]
        indice4 = np.where((np.argwhere(targetAmp)==[pointRow[-1],pointCol[0]]).all(axis=1))[0][0]

        angle_radian = angle*np.pi/180
        pointRow_rotated = np.round((pointCol-totalCol/2) * np.sin(angle_radian) + (pointRow-totalRow/2) * np.cos(angle_radian) + totalRow/2).astype(int)
        pointCol_rotated = np.round((pointCol-totalCol/2) * np.cos(angle_radian) - (pointRow-totalRow/2) * np.sin(angle_radian) + totalCol/2).astype(int)

        targetAmp_new[(pointRow_rotated, pointCol_rotated)] = targetAmp[point]

        vertices = [[pointRow_rotated[indice1],pointCol_rotated[indice1]], 
                    [pointRow_rotated[indice2],pointCol_rotated[indice2]],
                    [pointRow_rotated[indice3],pointCol_rotated[indice3]],
                    [pointRow_rotated[indice4],pointCol_rotated[indice4]]
                    ]
        
        if Plot:
            self.plot_target(targetAmp_new, vertices)

        return targetAmp_new



    def modify_targetAmp(self, targetAmp):
        '''
        This function receives an intensity pattern with uniform intensity on different lattice sites, it will
        output a target intensity pattern taking into account the finite diffraction efficiency.

        Currently, this function uses the theoretical value, will update by measurements if necessary.
        '''
        row = np.size(targetAmp, axis=0)
        col = np.size(targetAmp, axis=1)
        x = np.arange(row)
        y = np.arange(col)
        X, Y = np.meshgrid(x, y)
        
        center = np.array([(row-1)/2, (col-1)/2])
        
        d = np.sqrt(((X - center[0])*self.Focalpitchx)**2 + ((Y - center[1])*self.Focalpitchy)**2 )
        dmax = self.focallength / self.magnification * self.wavelength / self.pixelpitch/2
        diffrac_efficiency = (np.sinc(1/2*d/dmax))**2
        targetAmp_diffrac = np.divide(targetAmp,diffrac_efficiency)

        return targetAmp_diffrac
    

    def plot_target(self, targetAmp, vertices = None, figsize = (10,5)):
        '''
        This function plots the target location distrubution and spot distrubution.

        If the target is rotated, you must specify the vertices.
        '''
        if vertices is None:
            # only valid for rec lattice.
            point = np.where(targetAmp)
            startRow = point[0][0]
            endRow = point[0][-1]
            startCol = point[1][0]
            endCol = point[1][-1]
            vertices = [[startRow,startCol], [startRow,endCol], [endRow,endCol], [endRow,startCol]]

        plt.figure(figsize=figsize)

        plt.subplot(1,2,1)
        plt.imshow(np.zeros_like(targetAmp))
        ax = plt.gca()
        poly = Polygon(vertices, closed=True, edgecolor='r', facecolor=None)
        ax.add_patch(poly)
        for vertex in vertices:
            plt.text(vertex[0], vertex[1], f'({vertex[1]}, {vertex[0]})', color='red',fontsize=6)
        plt.title('target location distrubution')
        
        print(np.nonzero(targetAmp))
        plt.subplot(1,2,2)
        plt.scatter(np.where(targetAmp)[1],np.where(targetAmp)[0],s=5)#imshow以第一个指标为纵轴，第二个指标为横轴
        plt.gca().invert_yaxis()
        plt.title('target spot distrubution')

        plt.tight_layout()
        plt.savefig('data/target_distribution.png')
        plt.close()


    
    def target3D_generate(self, Lattice_type, layer_num, distance = None, spacing = None, arraysize = None, translate = None, rotate = None, angle = None, modify = None, Plot = False):

        if distance is None:
            distance = self.distance
        if spacing is None:
            spacing = self.spacing
        if arraysize is None:
            arraysize = self.arraysize
        if translate is None:
            translate = self.translate
        if rotate is None:
            rotate = self.rotate
        if angle is None:
            angle = self.angle
        if modify is None:
            modify = self.modify

        spacingx = np.round(spacing[0]/self.Focalpitchx)
        spacingy = np.round(spacing[1]/self.Focalpitchy)

        if Lattice_type == 'RecABStack':

            targetAmp1 = self.target_generate(Lattice_type='Rec', distance = distance, spacing = spacing, arraysize = arraysize, translate = translate, rotate = False, modify = modify, Plot = False)
            targetAmp2 = np.zeros_like(targetAmp1)
            targetAmp2[(np.round(np.where(targetAmp1)[0]+spacingy/2).astype('int'), np.round(np.where(targetAmp1)[1]+spacingx/2).astype('int'))] = targetAmp1[np.where(targetAmp1)]

            if rotate:
                targetAmp1 = self.rotate_targetAmp(targetAmp1,angle)
                targetAmp2 = self.rotate_targetAmp(targetAmp2,angle)
                print(f'You have rotated the target by angle {angle}.')

            targetAmp3D = np.stack([targetAmp1,targetAmp2]*int(np.ceil(layer_num/2)),axis=0)[:layer_num]
                
        elif Lattice_type == 'RecAAStack':

            targetAmp1 = self.target_generate(Lattice_type='Rec', distance = distance, spacing = spacing, arraysize = arraysize, translate = translate, rotate = rotate, modify = modify, Plot = False)
            targetAmp3D = np.stack([targetAmp1,]*layer_num,axis=0)

        else:
            raise('Lattice type not found!')
        
        if Plot:
            nonzero_indices = np.argwhere(targetAmp3D)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for index in nonzero_indices:
                i, j, k = index
                ax.scatter(i, j, k, c='C'+str(i))

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            plt.title('3D target spot distrubution')
            plt.show()

        return targetAmp3D

    
    
    

    def camera_Amp_generate(self, init_targetAmp, camera_intensity_array):
        '''
        This function receives a measured intensity array. This array is ordered by the following procedure:
        The array starts from the point most closed to the origin, then ordered row by row.
        Then it will output the corresponding foci amp corresponding to this input intensity array.

        Camera intensities are real; ``init_targetAmp`` may be real or complex
        (the magnitude is used to locate spots).
        '''
        init_mag = np.abs(init_targetAmp)
        camera_Amp = np.zeros_like(init_mag, dtype=np.float64)
        camera_Amp[np.where(init_mag > 0)] = np.sqrt(camera_intensity_array)
        return camera_Amp


    def target_adapt(self, init_targetAmp, cameraAmp):
        '''
        Adapt a target amplitude from a camera-intensity measurement.

        Operates in magnitude (measurement) space: ``init_targetAmp`` may be
        complex but only ``|init_targetAmp|`` is used.  Camera amplitude is
        always real.
        '''
        init_mag = np.abs(init_targetAmp)
        targetAmp = init_mag / np.sqrt(np.sum(init_mag ** 2))
        targetAmpmask = (init_mag > 0) * 1
        totalsites = np.count_nonzero(init_mag)

        cameraInt = np.square(cameraAmp) / np.sum(np.square(cameraAmp))
        cameraInt_avg = np.multiply(np.sum(cameraInt) / totalsites, targetAmpmask)

        targetAmp_adapt = np.multiply(
            np.sqrt(np.divide(
                cameraInt_avg, cameraInt,
                out=np.zeros_like(cameraInt_avg),
                where=cameraInt != 0,
            )),
            targetAmp,
        )

        return targetAmp_adapt

    
    def phase_to_screen(self, SLM_Phase):
        # This function is to crop the center pixel area to fit to the SLM screen
        # SLM_Phase is the phase image calculated by WGS
        # SLMResX, SLMResY retrieve the size of the SLM screen

        # X is column
        col = np.size(SLM_Phase, axis=1)
        # Y is row
        row = np.size(SLM_Phase, axis=0)

        centerX = col/2
        centerY = row/2

        SLMResX = self.SLMRes[0]
        SLMResY = self.SLMRes[1]
        SLMRes = np.min([SLMResX, SLMResY])

        startCol = centerX-round(SLMRes/2)
        endCol = centerX+round(SLMRes/2)
        startRow = centerY-round(SLMRes/2)
        endRow = centerY+round(SLMRes/2)

        SLM_IMG = SLM_Phase[int(startRow):int(endRow), :][:, int(startCol):int(endCol)]
       # location = [startRow, endRow, startCol, endCol]

        SLM_bit = np.around((SLM_IMG+np.pi)/(2*np.pi)*256).astype('uint8')
        # SLM_bit = SLM_IMG

        SLM_screen = np.zeros((int(SLMResY), int(SLMResX)))
        col_SLM_bit = np.size(SLM_bit, axis=1)
        row_SLM_bit = np.size(SLM_bit, axis=0)

        startRow_screen = SLMResY/2-round(row_SLM_bit/2)
        endRow_screen = SLMResY/2+round(row_SLM_bit/2)
        startCol_screen = SLMResX/2-round(col_SLM_bit/2)
        endCol_screen = SLMResX/2+round(col_SLM_bit/2)

        SLM_screen[int(startRow_screen):int(endRow_screen), :][:, int(startCol_screen):int(endCol_screen)] = SLM_bit

        return SLM_screen.astype('uint8')
        
    
    def target_and_phase_all_save(self, targetAmp, fftAmp, fftPhase, SLM_Phase, SLM_screen, SLM_screen_Corrected, adapt_times):
        path = self.path1+'/'+self.Time+'_'+str(self.arraysize[0])+'x'+str(self.arraysize[1])+'-'+str(self.spacing[0])+'x'+str(self.spacing[1])+'-dis'+str(int(self.distance))+'_adapt'+str(adapt_times)
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path+'/targetAmp.npy', targetAmp)
        np.save(path+'/fftAmp.npy', fftAmp)
        np.save(path+'/fftPhase.npy', fftPhase)
        np.save(path+'/SLM_Phase.npy', SLM_Phase)
        np.save(path+'/SLM_screen.npy', SLM_screen)
        np.save(path+'/SLM_screen_Corrected.npy', SLM_screen_Corrected)

        
    def target_and_phase_save(self, targetAmp, slm_phase, slm_screen_f_corrected, info, adapt_times):
        path = self.path1+'/'+self.Time+'_'+str(self.arraysize[0])+'x'+str(self.arraysize[1])+'-'+str(self.spacing[0])+'x'+str(self.spacing[1])+'-dis'+str(int(self.distance))+'-'+str(info)+'_adapt'+str(adapt_times)
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path+'/targetAmp.npy', targetAmp)
        np.save(path+'/slm_phase.npy', slm_phase)
        np.save(path+'/slm_screen_f_corrected', slm_screen_f_corrected)


    def file_path_generate_according_to_date_and_time(self, Date_user_defined = None, Time_user_defined = None):
        self.Date = time.strftime("%Y-%m-%d", time.localtime()) 
        self.Time = time.strftime("%H-%M-%S", time.localtime()) 
        if Date_user_defined is not None:
            self.Date = Date_user_defined
        if Time_user_defined is not None:
            self.Time = Time_user_defined
        self.path1 = 'PhaseImageForSLM/'+self.Date
        # self.path2 = 'CameraImage/'+self.Date
        if not os.path.exists(self.path1):
            os.makedirs(self.path1)
        # if not os.path.exists(self.path2):
        #     os.makedirs(self.path2)


    def fresnel_lens_phase_generate(self, shift_distance,x0,y0):
        '''
        the fresnel lens phase, see notion for more details.
        '''

        Xps, Yps = np.meshgrid(np.linspace(0, self.SLMRes[0], self.SLMRes[0]), np.linspace(0, self.SLMRes[1], self.SLMRes[1]))

        X = (Xps-x0)*self.pixelpitch
        Y = (Yps-y0)*self.pixelpitch

        # k = 2*np.pi/self.wavelength

        fresnel_lens_phase = np.mod(np.pi*(X**2+Y**2)*shift_distance/(self.wavelength*self.focallength**2)*self.magnification**2,2*np.pi)

        fresnel_lens_screen = np.round(fresnel_lens_phase/(2*np.pi)*256).astype('uint8')

        return fresnel_lens_screen, fresnel_lens_phase

    


    def phase_to_fftField(self, SLM_Phase):
        
        SLM_Field = np.multiply(self.initGaussianAmp, np.exp(1j*SLM_Phase))
        SLM_Field_shift = sp.fft.fftshift(SLM_Field)
        fftSLM = sp.fft.fft2(SLM_Field_shift)
        fftSLMShift = sp.fft.fftshift(fftSLM)
        fftSLM_norm = np.sqrt(np.sum(np.square(np.abs(fftSLMShift))))
        fftSLMShift_norm = fftSLMShift/fftSLM_norm

        fftAmp = np.abs(fftSLMShift_norm)
        fftPhase = np.angle(fftSLMShift_norm)
        return fftAmp, fftPhase
    

    def get_point_and_phase_list(self, targetAmp, fftPhase):

        '''
        input the targetAmp  and the fftPhase, 
        this function will pick up the position coordinates and and the corresponding phase of the tweezer arrays.
        
        '''
    
        point_list = np.flip(np.argwhere(targetAmp), axis=0)
        phase_list = [fftPhase[point[0],point[1]] for point in point_list]

        return point_list, phase_list
    

    def zernike_generate(self, zernike_aperture_radius=None, ind_Zernike_list=None, percent_list=None, isZernikePhaseContinous=None):
        '''
        this function will output zernike screen according to your input zernike indices and percents.
        '''
        # 该函数生成一个Zernike的screen图，注意此时范围是0-255
        
        if zernike_aperture_radius is None:
            zernike_aperture_radius = self.zernike_aperture_radius
        if ind_Zernike_list is None:
            ind_Zernike_list = self.ind_Zernike_list
        if percent_list is None:
            percent_list = self.percent_list
        if isZernikePhaseContinous is None:
            isZernikePhaseContinous = self.isZernikePhaseContinous
            

        SLM_aberr_screen_sum = 0
        for j in range(len(ind_Zernike_list)):
            myOberrationCorr = Zernike(self.SLMRes[0], self.SLMRes[1], self.pixelpitch, zernike_aperture_radius, ind_Zernike_list[j], percent_list[j])
            SLM_aberr_screen, m, n = myOberrationCorr.phase_Zernike(Plot = False, Save=False)
            if isZernikePhaseContinous:
                SLM_aberr_screen = myOberrationCorr.phase_Zernike_continuous(m, n, Plot = False)
            SLM_aberr_screen_sum  = SLM_aberr_screen_sum + SLM_aberr_screen
        plt.imshow(SLM_aberr_screen_sum)
        plt.colorbar()
        plt.show()
        # print(np.max(SLM_aberr_screen_sum))
        # print(np.min(SLM_aberr_screen_sum))

        # convert total phase between -pi and pi
        min_add = np.min(SLM_aberr_screen_sum)
        if min_add < 0:
            ind_2pi = np.abs(np.floor(np.divide(min_add, 2 * np.pi)))
            # print(ind_2pi)
            SLM_aberr_screen_sum = SLM_aberr_screen_sum + ind_2pi * 2 * np.pi
        SLM_aberr_screen_sum_mod = np.mod(SLM_aberr_screen_sum, 2 * np.pi)
        Zernike_Phase = np.multiply((SLM_aberr_screen_sum_mod <= np.pi), SLM_aberr_screen_sum_mod) \
                            + np.multiply((SLM_aberr_screen_sum_mod > np.pi),SLM_aberr_screen_sum_mod - 2 * np.pi)  # 让相位连续，此时范围为-pi到pi
        
        Zernike_screen = np.around((Zernike_Phase+np.pi)/(2*np.pi)*256).astype('uint8')

        return Zernike_screen


    # ------------------------------------------------------------------
    # CGM-only complex-target factories
    # ------------------------------------------------------------------
    # These wrap the free functions in ``slm.targets`` onto the instance's
    # (ImgResY, ImgResX) grid.  Each returns a complex128 field whose phase
    # is load-bearing: WGS would discard the phase and produce a bogus
    # intensity-only hologram, so these targets are CGM-only.

    def top_hat_target(self, radius, center=None):
        """CGM-only: circular flat-top target (uniform amplitude, flat phase)."""
        from slm.targets import top_hat
        return top_hat((int(self.ImgResY), int(self.ImgResX)), radius=radius, center=center)

    def gaussian_line_target(
        self, length, width_sigma, angle=0.0, center=None, phase_gradient=0.0,
    ):
        """CGM-only: 1D line with Gaussian cross-section and linear phase ramp."""
        from slm.targets import gaussian_line
        return gaussian_line(
            (int(self.ImgResY), int(self.ImgResX)),
            length=length,
            width_sigma=width_sigma,
            angle=angle,
            center=center,
            phase_gradient=phase_gradient,
        )

    def light_sheet_target(
        self, flat_width, gaussian_sigma, angle=0.0, center=None, edge_sigma=0.0,
        reweight=None, Plot=False,
    ):
        """CGM-only: 1D top-hat (Rydberg light sheet) with Gaussian perpendicular."""
        from slm.targets import light_sheet
        target_amp = light_sheet(
            (int(self.ImgResY), int(self.ImgResX)),
            flat_width=flat_width,
            gaussian_sigma=gaussian_sigma,
            angle=angle,
            center=center,
            edge_sigma=edge_sigma,
            reweight=reweight,
        )
        if Plot:
            amp = np.abs(target_amp)
            phase = np.angle(target_amp)
            # Auto-detect a compact zoom ROI from high-amplitude support.
            # This avoids huge boxes caused by very weak long tails.
            peak = np.max(amp)
            support = amp >= (peak * 0.05) if peak > 0 else np.zeros_like(amp, dtype=bool)
            if not np.any(support):
                support = amp > 0
            rows, cols = np.where(support)
            if rows.size > 0:
                margin = 10
                r0 = max(0, rows.min() - margin)
                r1 = min(amp.shape[0], rows.max() + margin + 1)
                c0 = max(0, cols.min() - margin)
                c1 = min(amp.shape[1], cols.max() + margin + 1)
                amp_zoom = amp[r0:r1, c0:c1]
                phase_zoom = phase[r0:r1, c0:c1]
                zoom_shape = amp_zoom.shape
            else:
                amp_zoom = amp
                phase_zoom = phase
                zoom_shape = amp.shape

            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.imshow(amp, cmap="magma")
            plt.colorbar()
            plt.title("light sheet target amplitude")
            plt.subplot(2, 2, 2)
            plt.imshow(phase, cmap="twilight")
            plt.colorbar()
            plt.title("light sheet target phase")
            plt.subplot(2, 2, 3)
            plt.imshow(amp_zoom, cmap="magma")
            plt.colorbar()
            plt.title(f"light sheet amplitude (zoom, {zoom_shape[0]}x{zoom_shape[1]})")
            plt.subplot(2, 2, 4)
            plt.imshow(phase_zoom, cmap="twilight")
            plt.colorbar()
            plt.title(f"light sheet phase (zoom, {zoom_shape[0]}x{zoom_shape[1]})")
            plt.tight_layout()
            plt.show()
        return target_amp

    def stationary_phase_sheet(
        self, flat_width, gaussian_sigma=None, angle=0.0, center=None,
    ):
        """Stationary-phase SLM seed for a light-sheet target.

        Analytic 1D top-hat phase (Gaussian -> flat-top along the rotated
        u-axis) from geometric-optics ray-density redistribution (see
        ``references/Top Hat Beam.pdf`` and ``slm.initial_phase``).  Pass the
        result as ``initSLMPhase`` to :func:`slm.cgm.CGM_phase_generate` to
        warm-start CGM for a :meth:`light_sheet_target` with matching
        arguments.

        When ``gaussian_sigma`` is given, a cylindrical Fresnel lens is added
        along the perpendicular axis to pre-broaden the natural focal-plane
        Gaussian ``lambda*f/(pi*w0)`` up to the target's perpendicular 1/e^2
        intensity radius (= ``gaussian_sigma * sqrt(2) * Focalpitchy``).  This
        makes the seed a 2D match for :meth:`light_sheet_target`.  Omit
        ``gaussian_sigma`` for pure 1D (along-line only) shaping.

        Parameters
        ----------
        flat_width : pixels (along-line top-hat full width, same units as
            :meth:`light_sheet_target`).
        gaussian_sigma : pixels; 1-sigma of the target amplitude Gaussian
            perpendicular to the line (matches
            :meth:`light_sheet_target.gaussian_sigma`).  ``None`` disables
            the cylindrical-lens term.
        angle : rotation of the along-line axis (radians, 0 = horizontal).
        center : (row, col) pixel-indexed focal-plane centre of the target,
            or ``None`` for the grid centre.  Same convention as
            :meth:`light_sheet_target`.
        """
        from slm.initial_phase import stationary_phase_light_sheet

        ny, nx = int(self.ImgResY), int(self.ImgResX)
        if center is None:
            center_um = (0.0, 0.0)
        else:
            row, col = center
            center_um = (
                (float(col) - (nx - 1) / 2.0) * float(self.Focalpitchx),
                (float(row) - (ny - 1) / 2.0) * float(self.Focalpitchy),
            )
        # The light_sheet target uses perp = exp(-v^2 / (2*sigma^2)), so the
        # amplitude has stddev ``sigma`` and the intensity 1/e^2 radius is
        # sigma*sqrt(2).  That's what the cylindrical-lens helper expects.
        if gaussian_sigma is None:
            perp_target_w_um = None
        else:
            perp_target_w_um = (
                float(gaussian_sigma) * np.sqrt(2.0) * float(self.Focalpitchy)
            )
        return stationary_phase_light_sheet(
            (ny, nx),
            flat_width_um=float(flat_width) * float(self.Focalpitchx),
            w0_um=float(self.beamwaist),
            wavelength_um=float(self.wavelength),
            focal_length_um=float(self.focallength) / float(self.magnification),
            pixel_pitch_um=float(self.pixelpitch),
            angle=float(angle),
            center_um=center_um,
            beam_center_um=tuple(getattr(self, "beam_center_um", (0.0, 0.0))),
            perp_target_w_um=perp_target_w_um,
        )

    def stationary_phase_sheet_1d(self, flat_width, center=None):
        """1D along-x stationary-phase seed for the 1D CGM path (issue #21).

        Thin wrapper: calls the shared 1D physics in
        :func:`slm.initial_phase.stationary_phase_light_sheet` with a
        degenerate ``(1, nx)`` grid and returns the single row, so no
        along-x math is duplicated.  ``angle=0`` is implicit (the 1D
        decomposition only makes sense for the x-axis).

        Parameters
        ----------
        flat_width : float
            Along-line top-hat width in focal-plane pixels (same units as
            :meth:`stationary_phase_sheet` and :meth:`light_sheet_target`).
        center : float | None
            Focal-plane column (px) to centre the target at, or ``None``
            for the grid centre.

        Returns
        -------
        1D float64 array of length ``self.ImgResX``, suitable as
        ``initSLMPhase`` to :func:`slm.cgm.CGM_phase_generate_1d`.
        """
        from slm.initial_phase import stationary_phase_light_sheet

        nx = int(self.ImgResX)
        if center is None:
            center_um = (0.0, 0.0)
        else:
            center_um = ((float(center) - (nx - 1) / 2.0) * float(self.Focalpitchx), 0.0)
        result_2d = stationary_phase_light_sheet(
            (1, nx),
            flat_width_um=float(flat_width) * float(self.Focalpitchx),
            w0_um=float(self.beamwaist),
            wavelength_um=float(self.wavelength),
            focal_length_um=float(self.focallength) / float(self.magnification),
            pixel_pitch_um=float(self.pixelpitch),
            angle=0.0,
            center_um=center_um,
            beam_center_um=tuple(getattr(self, "beam_center_um", (0.0, 0.0))),
            perp_target_w_um=None,
        )
        return np.asarray(result_2d[0, :], dtype=np.float64)

    def lg_mode_target(self, ell, p, w0, center=None):
        """CGM-only: Laguerre-Gaussian mode LG^p_ell with vortex phase."""
        from slm.targets import lg_mode
        return lg_mode((int(self.ImgResY), int(self.ImgResX)), ell=ell, p=p, w0=w0, center=center)

    def square_lattice_vortex_target(
        self, rows, cols, spacing, peak_sigma, ell=1, center=None,
    ):
        """CGM-only: square grid of Gaussian peaks with global vortex phase."""
        from slm.targets import square_lattice_vortex
        return square_lattice_vortex(
            (int(self.ImgResY), int(self.ImgResX)),
            rows=rows, cols=cols, spacing=spacing, peak_sigma=peak_sigma,
            ell=ell, center=center,
        )

    def ring_lattice_vortex_target(
        self, n_sites, ring_radius, peak_sigma, ell=1, center=None,
    ):
        """CGM-only: ring of Gaussian peaks with global vortex phase."""
        from slm.targets import ring_lattice_vortex
        return ring_lattice_vortex(
            (int(self.ImgResY), int(self.ImgResX)),
            n_sites=n_sites, ring_radius=ring_radius, peak_sigma=peak_sigma,
            ell=ell, center=center,
        )

    def graphene_lattice_target(self, rows, cols, spacing, peak_sigma, center=None):
        """CGM-only: honeycomb lattice with alternating 0/π phase on sublattices."""
        from slm.targets import graphene_lattice
        return graphene_lattice(
            (int(self.ImgResY), int(self.ImgResX)),
            rows=rows, cols=cols, spacing=spacing, peak_sigma=peak_sigma,
            center=center,
        )








 

