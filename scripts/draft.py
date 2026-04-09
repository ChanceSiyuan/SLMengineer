from slm.generation import SLM_class
import numpy as np

SLM = SLM_class()
SLM.image_init(initGaussianPhase_user_defined=np.zeros((4096, 4096)), Plot=False)
targetAmp = SLM.target_generate("Rec", arraysize=[4, 4], translate=True, Plot=False)

SLM.plot_target(targetAmp)
