#!/usr/bin/env python

import lal
from pycbc.waveform import TemplateBank
import pycbc.waveform as pw
import pycbc.filter as pf
import pycbc.types as pt

import pycbc.psd as pp
import pycbc.pnutils as pnu
from pycbc.pnutils import A0, A3
import copy
import scipy.optimize as so
import h5py
import numpy as np

# In[51]:

def mtotal_eta_to_tau0_tau3(m_total, eta, f_lower):
    m_total = m_total * lal.MTSUN_SI
    tau0 = A0(f_lower) / (m_total**(5./3.) * eta)
    tau3 = A3(f_lower) / (m_total**(2./3.) * eta)
    return tau0,tau3


def get_dm_for_tau3(dm, tau0, tau3, f_lower, deta):
    '''get dm so that dtau0 = 0 and only along tau3
    '''
    m, eta = pnu.tau0_tau3_to_mtotal_eta(tau0, tau3, f_lower)
    rt0, _ = mtotal_eta_to_tau0_tau3(m+dm, eta+deta, f_lower)
    return rt0 - tau0

def get_dm_for_tau0(dm, tau0, tau3, f_lower, deta):
    '''get dm so that dtau3 = 0 and only along tau0
    '''
    m, eta = pnu.tau0_tau3_to_mtotal_eta(tau0, tau3, f_lower)
    _, rt3 = mtotal_eta_to_tau0_tau3(m+dm, eta+deta, f_lower)
    return rt3 - tau3

def temp_tau0_tau3_with_valid_dtau0(tau0, tau3, f_ref):
    m, eta = pnu.tau0_tau3_to_mtotal_eta(tau0, tau3, f_ref)
    # We set delta_eta to 10% of eta
    deta = - 0.001 * eta
    # estimate using derivative and setting to zero
    dm3 = - 1.5 * m/eta * deta
    rdm3 = so.fsolve(get_dm_for_tau0, dm3, (tau0, tau3, f_ref, deta))[0]
    return mtotal_eta_to_tau0_tau3(m+rdm3, eta+deta, f_ref)

def temp_tau0_tau3_with_valid_dtau3(tau0, tau3, f_ref):
    m, eta = pnu.tau0_tau3_to_mtotal_eta(tau0, tau3, f_ref)
    # We set delta_eta to 10% of eta
    deta = - 0.001 * eta
    # estimate using derivative and setting to zero
    dm0 = 0.6 * m/eta * deta
    rdm0 = so.fsolve(get_dm_for_tau3, dm0, (tau0, tau3, f_ref, deta))[0]
    return mtotal_eta_to_tau0_tau3(m+rdm0, eta+deta, f_ref)

def temp_param_from_central_param(central_param, newtau0, newtau3, f_ref):

    temp_param = copy.deepcopy(central_param)
    m1, m2 = pnu.tau0_tau3_to_mass1_mass2(newtau0, newtau3, f_ref)
    temp_param['mass1'] = m1
    temp_param['mass2'] = m2
    temp_param['tau0'] = newtau0
    temp_param['tau3'] = newtau3
    temp_param['approximant'] = 'TaylorF2RedSpin'
    return temp_param

def get_chirp_time_region(trigger_params, psd, miss_match, f_lower=30., f_max=2048., f_ref=30.):
    central_param = copy.deepcopy(trigger_params)
    # if central_param['approximant'] == 'SPAtmplt':
    central_param['approximant'] == 'TaylorF2RedSpin'
    # if not ('tau0' and 'tau3' in central_param):
    #     t0, t3 = pnu.mass1_mass2_to_tau0_tau3(central_param['mass1'], central_param['mass2'], f_ref)
    # else:
    #     t0 = central_param['tau0']
    #     t3 = central_param['tau3']
    # for tau0 boundary
    newt0, newt3 = temp_tau0_tau3_with_valid_dtau0(central_param['tau0'], central_param['tau3'], f_ref)
    temp_param0 = temp_param_from_central_param(central_param, newt0, newt3, f_ref)
    # for tau3 boundary
    newt0, newt3 = temp_tau0_tau3_with_valid_dtau3(central_param['tau0'], central_param['tau3'], f_ref)
    temp_param3 = temp_param_from_central_param(central_param, newt0, newt3, f_ref)


    tlen = pnu.nearest_larger_binary_number(max([central_param['tau0'], temp_param0['tau0'], temp_param3['tau0']]))
    df = 1.0/tlen
    flen = int(f_max/df) + 1

    # hp = pt.zeros(flen, dtype=pt.complex64)
    # hp0 = pt.zeros(flen, dtype=pt.complex64)
    # hp3 = pt.zeros(flen, dtype=pt.complex64)

    # print central_param['approximant']

    # if central_param['approximant'] == 'SPAtmplt':
    #     central_param['approximant'] == 'TaylorF2RedSpin'
        # hp = pw.get_waveform_filter(hp, central_param, delta_f=df, f_lower=f_lower, f_ref=f_ref, f_final=f_max)
        # hp0 = pw.get_waveform_filter(hp0, temp_param0, delta_f=df, f_lower=f_lower, f_ref=f_ref, f_final=f_max)
        # hp3 = pw.get_waveform_filter(hp3, temp_param3, delta_f=df, f_lower=f_lower, f_ref=f_ref, f_final=f_max)
    # else:
    hp, hc = pw.get_fd_waveform(central_param, delta_f=df, f_lower=f_lower, f_ref=f_ref, f_final=f_max)
    hp0, hc0 = pw.get_fd_waveform(temp_param0, delta_f=df, f_lower=f_lower, f_ref=f_ref, f_final=f_max)
    hp3, hc3 = pw.get_fd_waveform(temp_param3, delta_f=df, f_lower=f_lower, f_ref=f_ref, f_final=f_max)


    # FIXME: currently will using aLIGOZeroDetHighPower
    # FIXME: add how to make sure, psd numerical problems of psd
    if psd is not None:
        ipsd = pp.interpolate(psd, df)
    else:
        ipsd = None
        # ipsd = pp.aLIGOZeroDetHighPower(flen, df, f_lower)
        # ipsd = pp.interpolate(ipsd, df)
        # ipsd.data[-1] = 2.0*ipsd.data[-2]
        # ipsd = ipsd.astype(hp.dtype)


    mat0, _ = pf.match(hp, hp0, ipsd, f_lower, f_max)
    mat3, _ = pf.match(hp, hp3, ipsd, f_lower, f_max)
    # print mat0, mat3, miss_match
#     print central_param['tau0'], central_param['tau3']
#     print temp_param0['tau0'], temp_param0['tau3']
#     print temp_param3['tau0'], temp_param3['tau3']
#     print float(temp_param0['tau0'])-float(central_param['tau0'])
#     print temp_param3['tau3']-central_param['tau3']
    dtau0_range = miss_match*(temp_param0['tau0']-central_param['tau0'])/(1.0-mat0)
    dtau3_range = miss_match*(temp_param3['tau3']-central_param['tau3'])/(1.0-mat3)
#     print dtau0_range, dtau3_range
    return dtau0_range, dtau3_range

def reduced_bank_for_signale_trigger(bank, trigger_param, psd, f_lower, f_max,
                                     f_ref, hierarchy_param, miss_match):
    if hierarchy_param == 'chirp_times':
        if not ('tau0' and 'tau3' in bank.parameters):
#             print "I am in"
            t0, t3 = pnu.mass1_mass2_to_tau0_tau3(bank.table['mass1'], bank.table['mass2'], f_ref)
            bank.table = bank.table.add_fields([t0, t3], ['tau0', 'tau3'])
        # print "calculating ranges..."
        hpsd = copy.deepcopy(psd)
        hpsd = hpsd.astype(pt.float64)
        dtau0_range, dtau3_range = get_chirp_time_region(trigger_param, hpsd, miss_match, f_lower, f_max, f_ref)
        reqd_idx = []

        reqd_idx = abs(bank.table['tau0'] - trigger_param['tau0']) <= 2.0*abs(dtau0_range)
        reqd_idx *= (abs(bank.table['tau3'] - trigger_param['tau3']) <= 2.0*abs(dtau3_range))
        newbank = copy.copy(bank)
        newbank.table = newbank.table[reqd_idx]

        while len(newbank.table) < 4 or len(newbank.table) > 350:
            if len(newbank.table) < 4:
                dtau0_range *= 1.5
                dtau3_range *= 1.5
                reqd_idx = abs(bank.table['tau0'] - trigger_param['tau0']) <= 2.0*abs(dtau0_range)
                reqd_idx *= (abs(bank.table['tau3'] - trigger_param['tau3']) <= 2.0*abs(dtau3_range))
                newbank = copy.copy(bank)
                newbank.table = newbank.table[reqd_idx]
            elif len(newbank.table) > 350:
                dtau0_range /= 2.0
                dtau3_range /= 2.0
                reqd_idx = abs(bank.table['tau0'] - trigger_param['tau0']) <= 2.0*abs(dtau0_range)
                reqd_idx *= (abs(bank.table['tau3'] - trigger_param['tau3']) <= 2.0*abs(dtau3_range))
                newbank = copy.copy(bank)
                newbank.table = newbank.table[reqd_idx]

            break
        # print 'sngl trig bank:', len(newbank.table)

        return newbank, reqd_idx

def get_seg_triggers(trigger_file, det, bank, seg_start, seg_end, f_ref=30.0):
#     print trigger_file
    f = h5py.File(trigger_file)
    trig_snr = f['{0}/snr'.format(det)][...]
    idx = (trig_snr > 6.0)
    trigger_hashes = f['{0}/template_hash'.format(det)][...][idx]
    trigger_end_times = f['{0}/end_time'.format(det)][...][idx]
    f.close()
    print seg_start, seg_end
    rel_idx = np.where(seg_start < trigger_end_times) and (trigger_end_times < seg_end)
    rel_hashes = set(trigger_hashes[rel_idx])
    print trigger_end_times[rel_idx][:].min(), trigger_end_times[rel_idx][:].max()
    print "length of the trigger list for seg start-end:", len(rel_hashes), seg_start, seg_end
    trigger_params = bank.table[np.in1d(bank.table['template_hash'], list(rel_hashes), True)]
    if not ('tau0' and 'tau3' in trigger_params):
        t0, t3 = pnu.mass1_mass2_to_tau0_tau3(trigger_params['mass1'], trigger_params['mass2'], f_ref)
        trigger_params = trigger_params.add_fields([t0, t3], ['tau0', 'tau3'])

    return trigger_params

def reduced_bank_for_segment(fine_bank, coarse_bank, trigger_file, det, seg_start, seg_end,
                             psd=None, f_lower=30.0, f_max=2048.0, f_ref=30.,
                             hierarchy_param='chirp_times', miss_match=0.1):

    seg_trig_par = get_seg_triggers(trigger_file, det, coarse_bank, seg_start, seg_end, f_ref)
    all_reqd_idx = []
    all_reqd_idx = np.array([False]*len(fine_bank.table['template_hash']))

    for trig_p in seg_trig_par:
        # print 'trig param:', trig_p
        _, idx = reduced_bank_for_signale_trigger(fine_bank, trig_p, psd, f_lower, f_max,
                                                  f_ref, hierarchy_param, miss_match)
        # print idx
        all_reqd_idx = np.logical_or(all_reqd_idx, idx)

    # print all_reqd_idx
    newbank = copy.copy(fine_bank)
    newbank.table = newbank.table[all_reqd_idx]
    return newbank
