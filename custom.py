 from utils.rtsys import SigIntercept

from utils.cleegn import CLEEGN
from utils.Siamese import Siamese_SCC
from torch import from_numpy as np2TT
from torchinfo import summary
import torch

from scipy import signal
import numpy as np
import math
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PreProcessing(SigIntercept):

    def __init__(self, lowcut, highcut, fsIn, fsOut, parent):
        super(PreProcessing, self).__init__(
            parent.bufShape[0], raiseStream=True, name="Pre-pocessedEEG", sfreq=fsOut,
            parent=parent
        )
        
        self.sos = self.__butter_bandpass(lowcut, highcut, fsIn, order=5)
        self.zi = signal.sosfilt_zi(self.sos)
        self.zi = np.repeat(np.expand_dims(self.zi, axis=1), self.bufShape[0], axis=1)
        self.step = math.ceil(fsIn / fsOut)
        self.a0 = 0  # next start index, downsample issue

    def __butter_bandpass(self, lowcut, highcut, fs, order=3):
        sos = signal.butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')
        return sos

    def befed(self):
        xLen = self.parent.latestDataLen
        x = self.parent.buffer[:, -xLen:].copy()
        t = self.parent.tstmps[-xLen:].copy()

        """ Common Average Re-reference (CAR) """
        x -= x.mean(axis=0)

        """ Band-pass filtering """
        x, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)

        # """ Downsample """
        indices = np.arange(self.a0, x.shape[1], dtype=int)[::self.step]
        x, t = x[:, indices], t[indices]

        self.a0 = indices[-1] + self.step - xLen
        return x, t



class CLEEGNing(SigIntercept):

    def __init__(self, model_path, fsOut, parent):
        super(CLEEGNing, self).__init__(
            parent.bufShape[0], raiseStream=True, name="CLEEGNedEEG", sfreq=fsOut,
            parent=parent
        )
        # Done: auto get n_channel, win_size from load_model, next update
        state = torch.load(model_path, map_location="cpu")
        self.model = CLEEGN(n_chan=4, fs=250.0, N_F=4).to(device)
        self.model.load_state_dict(state["state_dict"])
        # summary(model, input_size=(64, 1, 8, 512))

        self.tstmps = self.parent.tstmps  # point to parent tstmps
        self.flags = self.parent.flags  # point to parent flags

    def update(self):  # overridding `SigIntercept` update()
        self.model.eval()

        xLen, n_channel = 1000, self.bufShape[0]
        x = self.parent.buffer[:, -xLen:].copy()

        x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
        x = np2TT(x).to(device, dtype=torch.float)
        x = self.model(x)
        x = x.view(n_channel, -1).detach().cpu().numpy()
        self.buffer[:, -xLen:] = x

        #print(self.tstmps[-1], self.flags[-1])
        # return np.zeros((8, 0)), 0  # special design, skip update

class DrowsinessEst(SigIntercept):

    def __init__(self, model_path, fsout, parent):
        super(DrowsinessEst, self).__init__(parent.bufShape[0], raiseStream=False, name=' ', sfreq=fsout, parent=parent)

        self.model = Siamese_SCC(EEG_ch = 4, num_smooth = 10).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        
        self.tstmps = self.parent.tstmps  # point to parent tstmps
        self.flags = self.parent.flags  # point to parent flags

        self.baseline = np.zeros((10, 4, 750), dtype=np.float32)
        self.current = np.zeros((10, 4, 750), dtype=np.float32)
        self.delta_DI = 0
        self.indicator = 0 
        self.baseline_idx = 1
        self.startT = time.time()
    
    def update(self, x=None, t=None):

        xLen, n_channel = 750, self.bufShape[0]
        x = self.parent.buffer[:, -xLen:].copy()
        self.current[1:, :, :] = self.current[:-1, :, :]
        self.current[0, :, :] = x

        if self.indicator < 20:
            self.baseline = self.current.copy()

        currentT = time.time()
        if currentT - self.startT > 9 and baseline_idx <= 9:
            self.baseline[9-self.baseline_idx, :, :] = x
            self.startT = currentT
            baseline_idx += 1
        
        self.model.eval()
        x1 = np2TT(self.baseline).to(device, dtype=torch.float)    
        x2 = np2TT(self.current).to(device, dtype=torch.float)
        inputData = torch.cat([x2, x1], 0).unsqueeze(0)
        _, _, self.delta_DI = self.model(inputData)
        self.delta_DI = self.delta_DI[0,0].detach().cpu().numpy()

        self.indicator += 1

    def show(self):
        
        print('Drowsiness index: %f'%(abs(self.delta_DI)))