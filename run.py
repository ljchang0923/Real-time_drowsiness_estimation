from inspect import Parameter
from utils.decos import StreamInfo_
from utils.rtsys import SigIntercept
from utils.rtsys import BasicRecv

from custom import PreProcessing
from custom import CLEEGNing
from custom import DrowsinessEst

import prettytable as pt
import numpy as np
import pylsl
import math
import time
import os


def main():
    print("looking for an EEG stream...")
    streamList = pylsl.resolve_stream("type", "EEG")
    streamList = [StreamInfo_(s) for s in streamList]

    tb = pt.PrettyTable()
    tb.field_names = ["sid", "name", "type", "#_channel", "srate", "srcID"]
    for sid, stm in enumerate(streamList):
        sinfo = [sid, stm.name, stm.type, stm.n_chan, stm.srate, stm.ssid]
        tb.add_row(sinfo)
        # print(stm)
    print(tb)
    streamID = int(input("Select steam... "))
    selcStream = streamList[streamID]
    inlet = pylsl.StreamInlet(selcStream.lsl_stream())


    root = BasicRecv(4, selcStream.srate)
    """ temporary use local var, module it in future """
    block1 = PreProcessing(2, 30, selcStream.srate, 250.0, parent=root)
    block2 = CLEEGNing("drowsiness-4ch.pth", fsOut=250.0, parent=block1)
    block3 = DrowsinessEst("s44_070325n_model.pt", fsout=250.0, parent=block2)
    recived_ch = [0,1,6,7]
    # iteration = 1
    # acc_time = 0
    while True:
        pull_kwargs = {"timeout": 1, "max_samples": 256}
        chunk, timestamps = inlet.pull_chunk(**pull_kwargs)
        chunk = np.asarray(chunk, dtype=np.float32).T
        chunk = chunk[recived_ch,:]
        timestamps = np.asarray(timestamps, dtype=np.float32)
        if not len(timestamps):
            print(f"[x] Loss conection to the stream: {selcStream.name()}...")
            break # TODO: try recovery???

        root.update(chunk, timestamps)
        block1.update()
        block2.update()
        # start_t = time.time()
        block3.update()
        # end_t = time.time()

        block1.send(delay=2)
        block2.send(delay=2)
        block3.show()
        # acc_time += end_t - start_t
        # print(f"Inference time: {acc_time/iteration}")
        # iteration +=1


if __name__ == "__main__":
    # try:
    #     main()
    # except KeyboardInterrupt:
    #     print()
    #     exit(0)
    main()
