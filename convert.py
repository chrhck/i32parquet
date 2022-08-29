#!/usr/bin/env python
import os
from icecube import dataio, dataclasses, simclasses, recclasses, spline_reco
from icecube import lilliput, gulliver, paraboloid, millipede
import awkward as ak
import argparse


def convert_i3particle(part, frame):
    return {
        "type": int(part.type),
        "zenith": part.dir.zenith,
        "azimuth": part.dir.azimuth,
        "energy": part.energy,
        "x": part.pos.x,
        "y": part.pos.y,
        "z": part.pos.z,
        "time": part.time,
    }


def convert_i3vectori3particle(parts, frame):
    return [convert_i3particle(part, frame) for part in parts]


def convert_i3recopulseseriesmap(pulses, frame):
    try:
        pulses = pulses.apply(frame)
    except AttributeError:
        # pulses is not a MapMask, no need to call apply
        pass
    
    if len(pulses) == 0:
        return []

    hits = [] 
    for omkey, om_pulses in pulses.items():
        om_idx = omkey.om - 1
        string_idx = omkey.string - 1

        sensor_idx = string_idx * 60 + om_idx

        for i, pulse in enumerate(om_pulses):
            hits.append(
                {"sensor_idx": sensor_idx,
                 "time": pulse.time,
                 "charge": pulse.charge,
                 "flag": pulse.flags == 4 or pulse.flags == 5,
                })

    return hits


def convert_track_characteristics(char, frame):
    return {
        "avg_dom_dist_q_tot_dom": char.avg_dom_dist_q_tot_dom,
        "empty_hits_track_length": char.empty_hits_track_length,
        "track_hits_distribution_smoothness": char.track_hits_distribution_smoothness,
        "track_hits_separation_length": char.track_hits_separation_length
    }


def convert_directhit_characteristics(char, frame):
    return {
        "dir_track_hit_distribution_smoothness": char.dir_track_hit_distribution_smoothness,
        "dir_track_length": char.dir_track_length,
        "n_dir_doms": char.n_dir_doms,
        "n_dir_pulses": char.n_dir_pulses,
        "n_dir_strings": char.n_dir_strings,
        "n_early_doms": char.n_early_doms,
        "n_early_pulses": char.n_early_pulses,
        "n_early_strings": char.n_early_strings,
        "n_late_doms": char.n_late_doms,
        "n_late_pulses": char.n_late_pulses,
        "n_late_strings": char.n_late_strings,
        "q_dir_pulses": char.q_dir_pulses,
        "q_early_pulses": char.q_early_pulses,
        "q_late_pulses": char.q_late_pulses,
    }


def convert_i3geometry(geo, _):
    data = []
    for omkey, omg in geo.omgeo:
        data.append({
            "string": omkey.string,
            "om": omkey.om,
            "pos_x": omg.position.x,
            "pos_y": omg.position.y,
            "pos_z": omg.position.z,})
    return data


CONVERTERS = {
    # dataclasses.I3MCTree: convert_mctree,
    dataclasses.I3Particle: convert_i3particle,
    dataclasses.I3RecoPulseSeriesMap: convert_i3recopulseseriesmap,
    dataclasses.I3RecoPulseSeriesMapMask: convert_i3recopulseseriesmap,
    recclasses.I3DirectHitsValues: convert_directhit_characteristics,
    recclasses.I3TrackCharacteristicsValues: convert_track_characteristics,
    dataclasses.I3VectorI3Particle: convert_i3vectori3particle
}


def convert_file(fname, allowlist=None, outdir=None):
    file = dataio.I3File(fname)
    data = []

    geodata = None
    while file.more():
        fr = file.pop_frame()
        if "I3Geometry" in fr:
            geodata = convert_i3geometry(fr["I3Geometry"], fr)
            break
    
    while file.more():
        try:
            #frame = file.pop_daq()
            frame = file.pop_physics()
        except:
            break

        conv = {}
        for key in frame.keys():
            if (allowlist is not None) and (key not in allowlist):
                continue

            try:
                obj = frame[key]
            except KeyError:
                continue
            if type(obj) in CONVERTERS:
                conv[key] = CONVERTERS[type(obj)](obj, frame)
        data.append(conv)

    data = ak.from_iter(data)
    geodata = ak.Array(geodata)

    if outdir:
        outname = os.path.join(outdir, os.path.basename(fname))
    else:
        outname = fname

    ak.to_parquet(data, outname + ".parquet")
    ak.to_parquet(geodata, outname + ".geo.parquet")
    return data, geodata


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=None)
    parser.add_argument("files", nargs='*')
    args = parser.parse_args()

    allowlist = [
        "InIceDSTPulses", "SplineMPE", "SplineMPECharacteristics",
        "SplineMPEDirectHitsD", "SplineMPEIC_MillipedeHighEnergyMIE"]

    for fname in args.files:
        assert os.path.exists(fname)
        convert_file(fname, allowlist, args.outdir)
