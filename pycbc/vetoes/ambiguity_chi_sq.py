
# coding: utf-8

# In[1]:

import numpy as np
import numpy
from numpy import linalg as la
# import pylab as plt

import itertools
import copy
import logging

import pycbc.waveform as pw
import pycbc.filter as pf
import pycbc.psd as pp
from pycbc.types import TimeSeries, FrequencySeries, zeros, float32, complex64, complex128
import pycbc.types as pt
import pycbc.pnutils as pnu
import pycbc.noise as pn
from pycbc.fft import fft, ifft, IFFT


# In[3]:

_snr = None
def inner(vec1, vec2, psd=None,
          low_frequency_cutoff=None, high_frequency_cutoff=None,
          v1_norm=None, v2_norm=None):

    htilde = pf.make_frequency_series(vec1)
    stilde = pf.make_frequency_series(vec2)

    N = (len(htilde)-1) * 2

    global _snr
    _snr = None
    if _snr is None or _snr.dtype != htilde.dtype or len(_snr) != N:
        _snr = pt.zeros(N,dtype=pt.complex_same_precision_as(vec1))
        snr, corr, snr_norm = pf.matched_filter_core(htilde,stilde,psd,low_frequency_cutoff,
                                                  high_frequency_cutoff, v1_norm, out=_snr)
        # print snr, corr, snr_norm
    if v2_norm is None:
        v2_norm = pf.sigmasq(stilde, psd, low_frequency_cutoff, high_frequency_cutoff)

    snr.data = snr.data * snr_norm / np.sqrt(v2_norm)

    return snr


def segment_snrs(filters, stilde, psd, low_frequency_cutoff):
    """ This functions calculates the snr of each bank veto template against
    the segment

    Parameters
    ----------
    filters: list of FrequencySeries
        The list of bank veto templates filters.
    stilde: FrequencySeries
        The current segment of data.
    psd: FrequencySeries
    low_frequency_cutoff: float

    Returns
    -------
    snr (list): List of snr time series.
    """
    snrs = []
    for i, template in enumerate(filters):
        snr = inner(template, stilde, psd, low_frequency_cutoff, v2_norm=1.0)

        snrs.append(snr)
        print "lens of snr and stilde:"
        print len(snr), len(stilde)

    return snrs


def get_template_list(filters, trigger_template, f_low, n):
    bank_template_tau0, _ = pnu.mass1_mass2_to_tau0_tau3(filters.table['mass1'],
                                                    filters.table['mass2'], f_low)
    trigger_template_tau0, _ = pnu.mass1_mass2_to_tau0_tau3(trigger_template.params.mass1,
                                                        trigger_template.params.mass2, f_low)
    idx = np.argmin(abs(bank_template_tau0 - trigger_template_tau0))
    # ids = numpy.arange(-2*n, 2*n+2, 2) + idx
    ids = np.arange(-n, n+1, 1) + idx
    ids = ids[np.where((ids>=0) & (ids<len(filters.table)) & (ids!=idx))]
    trig_bank = copy.copy(filters)
    trig_bank.table = trig_bank.table[ids]
    return trig_bank


# In[ ]:

class SingleDetAmbiguityChiSq(object):
    def __init__(self, bank_file, flen, delta_f, f_low, cdtype, time_indices=[0], approximant=None, **kwds):
        if bank_file is not None:
            self.do = True

            self.column_name = "bank_chisq"
            self.table_dof_name = "bank_chisq_dof"

            self.cdtype = cdtype
            self.delta_f = delta_f
            self.f_low = f_low
            self.seg_len_freq = flen
            self.seg_len_time = (self.seg_len_freq-1)*2
            self.time_idices = np.array(time_indices)

            logging.info("Read in bank veto template bank")
            bank_veto_bank = pw.FilterBank(bank_file,
                    self.seg_len_freq,
                    self.delta_f,
                    dtype=self.cdtype,
                    low_frequency_cutoff=self.f_low,
                    approximant=approximant, **kwds)

            self.filters = bank_veto_bank

            self._relevent_filters = {}
            self._filter_matches_cache = {}
            self._segment_snrs_cache = {}
            self._sigma_cache = {}
#             self._template_filter_matches_cache = {}
        else:
            self.do = False

    def cache_segment_snrs(self, template, stilde, psd):
        key = (template.params.template_hash, hash(stilde), hash(psd))
        if key not in self._segment_snrs_cache:
            mat = inner(template, stilde*psd, psd, self.f_low, v2_norm=1.0)
            self._segment_snrs_cache[key] = mat

            # print "lens of snr and stilde:"
            # print len(mat), len(stilde)

        # relevant_filters = self.cache_filters(template, n)
        # if key not in self._segment_snrs_cache:
        #     mat = inner(template, stilde*psd, psd, self.f_low, v2_norm=1.0)
        #     self._segment_snrs_cache[key] = mat

        return self._segment_snrs_cache[key]

    def cache_filters(self, template, n):
        key = (template.params.template_hash, n)
        if key not in self._relevent_filters:
            self._relevent_filters[key] = get_template_list(self.filters, template,self.f_low, n)

        return self._relevent_filters[key]

    def cache_filter_matches(self, filter1, filter2, psd):
        key = (filter1.params.template_hash, filter2.params.template_hash, hash(psd))
        if key not in self._filter_matches_cache:
            mat = inner(filter1, filter2, psd, self.f_low)
            # print mat[0]
            self._filter_matches_cache[key] = mat
        return self._filter_matches_cache[key]

    def calculate_sigma(self, template, psd, n):
        key = (template.params.template_hash, n, hash(psd))
        if key not in self._sigma_cache:
            rel_filters = self.cache_filters(template, n)
            m = len(rel_filters)  # into len(time_indices)
            cov = np.ones((m, m))
            h_ij = np.ones((m, m))

            for i, j in itertools.combinations_with_replacement(range(m), 2):
                mat = self.cache_filter_matches(rel_filters[i], rel_filters[j], psd)
                mat_i_temp = self.cache_filter_matches(template, rel_filters[i], psd)
                mat_j_temp = self.cache_filter_matches(template, rel_filters[j], psd)
                h_ij[i, j] = mat.data[0].real
                h_ij[j, i] = h_ij[i, j]

                # print i, 'H0i:', mat_i_temp.data.real[0]
                # print 'H0j:', mat_j_temp.data.real[0]

                # print i, 'Hpii:', mat_i_temp.data.imag[0]
                # print 'H0j:', mat_j_temp.data.real[0]
                cov[i, j] = h_ij[i, j] - (mat_i_temp.data[0].real*mat_j_temp.data[0].real + mat_i_temp.data[0].imag*mat_j_temp.data[0].imag)
                cov[j, i] = cov[i, j]
            # for i, j in itertools.combinations(range(m), 2):
            #     mat = self.cache_filter_matches(rel_filters[i], rel_filters[j], psd)
            #     cov[i, j] = mat.data[0].real # replace with time_indices
            #     cov[j, i] = cov[i, j]

            evals, rot_mat = la.eig(cov)
            print "cov matrix", cov
            print '\n egainvals:', evals

            self._sigma_cache[key] = cov, evals, rot_mat

        # print 'h_ij:', h_ij
        return self._sigma_cache[key]

    def values(self, template, stilde, psd, snr, indices, n):
        """
        Returns
        -------
        bank_chisq_from_filters: TimeSeries of bank veto values - if indices
        is None then evaluated at all time samples, if not then only at
        requested sample indices

        bank_chisq_dof: int, approx number of statistical degrees of freedom
        """
        if self.do:
            chisq = []
            chisq_dof = []
            sigma, evals, rot_mat = self.calculate_sigma(template, psd, n)
            rel_filters = self.cache_filters(template, n)

            time_index = 0
            for j, idx in enumerate(indices):
                temp = np.zeros_like(evals)
                for i, fil in enumerate(rel_filters):
                    # temp_seg_snr = self.cache_segment_snrs(template, stilde, psd).data[idx]
                    # print temp_seg_snr
                    # print len(temp_seg_snr)
                    # temp_fil_mat = self.cache_filter_matches(template, fil, psd).data[0]

                    # temp[i] = (self.cache_segment_snrs(template, stilde, psd).data[idx+time_index] - \
                    #         snr[j]*self.cache_filter_matches(template, fil, psd).data[0+time_index]).real
                    proj =  self.cache_filter_matches(template, fil, psd).data[0]
                    temp[i] = self.cache_segment_snrs(fil, stilde, psd).data[idx+time_index].real - proj.real*snr[j].real - proj.imag*snr[j].imag

                # print "del_ci", temp
                temp = np.dot(rot_mat.T, temp)
                # print "del_ci, prime", temp
                idx = evals > 0.05
                chi = (temp[idx]*temp[idx]/evals[idx]).sum()

                print "snr, chi_sq, dof: ", snr[j], chi, len(evals[idx]), '\n'
                chisq.append(chi) # assuming all eigenvalues are good
                chisq_dof.append(len(evals))

            return chisq, chisq_dof

        else:
            return None, None

    def save_cav_mat_eval(self, filename):
        np.save(filename, self._sigma_cache)
        logging.info("Wriiten cov matrix.")



