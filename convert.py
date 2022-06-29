#!/usr/bin/env python
import os, sys
from icecube import dataio
import numpy as np
import awkward as ak
import argparse

#geo = np.load('geo_array.npy')

def convert(fname, outdir=None):
    file = dataio.I3File(fname)
    data = []

    while file.more():
        try:
            #frame = file.pop_daq()
            frame = file.pop_physics()
        except:
            break

        try:
            pulses = frame['SplitInIcePulses']
        except KeyError:
            continue
        try:
            pulses = pulses.apply(frame)
        except AttributeError:
            # pulses is not a MapMask, no need to call apply
            pass
        
        if len(pulses) == 0:
            continue

        hits = [] 
        for omkey, om_pulses in pulses.items():
            om_idx = omkey.om - 1
            string_idx = omkey.string - 1

            sensor_idx = string_idx * 60 + om_idx

            #xyz = geo[string_idx, om_idx]

            for i, pulse in enumerate(om_pulses):
                hits.append({'sensor_idx':sensor_idx,
                             #'string_idx':string_idx,
			     #'dom_idx':om_idx,
			     #'x':xyz[0],
                             #'y':xyz[1],
			     #'z':xyz[2],
			     'time':pulse.time,
			     'charge':pulse.charge,
			     'flag':pulse.flags == 4 or pulse.flags == 5,
			     })


        MC = frame['I3MCTree']
        data.append({'features':hits, 'labels':{'azimuth':MC[0].dir.azimuth, 'zenith':MC[0].dir.zenith}})

    if len(data) == 0:
        return
    a = ak.from_iter(data)

    fname = fname.rstrip('i3.zst')+'.parquet'

    if outdir:
        fname = os.path.join(outdir, os.path.split(fname)[1])

    print(f"Writing to {fname}")
    ak.to_parquet(a, fname)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=None)
    parser.add_argument("files", nargs='*')
    args = parser.parse_args()

    for fname in args.files: 
        assert os.path.exists(fname)
        print(f"Opening {fname}")
        convert(fname, outdir=args.outdir)
