import tools.use_denoiser as d
import tools.compute_metrics_no_GT as cm
import tools.use_AE_db as ae
import matplotlib.pyplot as plt
import tol_colors as tc
import pandas as pd


def results(subject, start_min, end_min):
    ecg_sig = ae.read_sub_ecg(subject, i=int(360 * 60 * start_min), f=int(360 * 60 * end_min))

    sig = d.prepare_ecg(ecg_sig, 350)

    model = d.import_model()

    title = 'Subject ' + str(subject) + ' - i: min ' + str(start_min) + '; f: min ' + str(end_min)
    title_im = 'Subject_' + str(subject) + '_i-min' + str(start_min) + '_f-min' + str(end_min)

    clean = d.clean_ecg(sig, model, title=title + '. No segmentation', figures=False)
    sig_clean = clean[0]
    time_no_seg = clean[1]
    print('Time (no segmentation): ' + str(clean[1]))

    clean_seg_align = d.clean_ecg_segments(sig, model, title=title + '. With segmentation (with post alignment)',
                                           postalign=True, figures=False)
    sig_clean_seg_align = clean_seg_align[0]
    time_seg_align = clean_seg_align[1]
    print('Time (with segmentation): ' + str(clean_seg_align[1]))

    sig = sig.flatten()

    # before cleaning
    up_lim, low_lim = cm.outliers_limits(sig)
    peaks_ind_bef, rr_dist_bef, [out_up_bef, out_low_bef] = cm.rr_outliers(sig, up_lim, low_lim, plots=False)
    miss_bef = cm.missing_peaks(rr_dist_bef, out_up_bef)
    wron_bef = cm.wrong_detection(out_low_bef)
    print('Before: ' + str(miss_bef) + ' missing and ' + str(wron_bef) + ' wrongly detected')
    # after cleaning
    peaks_ind_aft_1, rr_dist_aft_1, [out_up_aft_1, out_low_aft_1] = cm.rr_outliers(sig_clean, up_lim, low_lim,
                                                                                   title=title, plots=False)
    miss_aft_1 = cm.missing_peaks(rr_dist_aft_1, out_up_aft_1)
    wron_aft_1 = cm.wrong_detection(out_low_aft_1)
    print('After (without segmentation): ' + str(miss_aft_1) + ' missing and ' + str(wron_aft_1) + ' wrongly detected')
    peaks_ind_aft_3, rr_dist_aft_3, [out_up_aft_3, out_low_aft_3] = cm.rr_outliers(sig_clean_seg_align, up_lim, low_lim,
                                                                                   title=title, plots=False)
    miss_aft_3 = cm.missing_peaks(rr_dist_aft_3, out_up_aft_3)
    wron_aft_3 = cm.wrong_detection(out_low_aft_3)
    print('After (with segmentation): ' + str(miss_aft_3) + ' missing and ' + str(wron_aft_3) + ' wrongly detected')

    colors = list(tc.tol_cset('bright'))
    plt.figure()
    plt.plot(sig, c=colors[-1], label='before')
    plt.plot(sig_clean, c=colors[-2], label='denoised (no seg)')
    plt.plot(sig_clean_seg_align, c=colors[1], label='denoised (seg and align)')
    plt.legend()
    plt.savefig('results/Factory/' + 'ECG_signals_comparison-' + title_im)

    plt.figure()
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(sig, c=colors[-1])
    ax1.plot(peaks_ind_bef, sig[peaks_ind_bef], 'o', c=colors[-2], label='peaks')
    ax1.plot(peaks_ind_bef[out_up_bef], sig[peaks_ind_bef[out_up_bef]], '*', c=colors[0], label='outlier - out_up')
    ax1.plot(peaks_ind_bef[out_low_bef], sig[peaks_ind_bef[out_low_bef]], '*', c=colors[1], label='outlier -out_low')
    ax1.axes.get_xaxis().set_visible(False)
    plt.title('Before')
    ax1.legend()
    ax2 = plt.subplot(3, 1, 2, sharex=ax1, sharey=ax1)
    ax2.plot(sig_clean, c=colors[-1])
    ax2.plot(peaks_ind_aft_1, sig_clean[peaks_ind_aft_1], 'o', c=colors[-2])
    ax2.plot(peaks_ind_aft_1[out_up_aft_1], sig_clean[peaks_ind_aft_1][out_up_aft_1], '*', c=colors[0])
    ax2.plot(peaks_ind_aft_1[out_low_aft_1], sig_clean[peaks_ind_aft_1][out_low_aft_1], '*', c=colors[1])
    ax2.axes.get_xaxis().set_visible(False)
    plt.title('After - no segmentation')
    ax3 = plt.subplot(3, 1, 3, sharex=ax1, sharey=ax1)
    ax3.plot(sig_clean_seg_align, c=colors[-1])
    ax3.plot(peaks_ind_aft_3, sig_clean_seg_align[peaks_ind_aft_3], 'o', c=colors[-2])
    ax3.plot(peaks_ind_aft_3[out_up_aft_3], sig_clean_seg_align[peaks_ind_aft_3][out_up_aft_3], '*', c=colors[0])
    ax3.plot(peaks_ind_aft_3[out_low_aft_3], sig_clean_seg_align[peaks_ind_aft_3][out_low_aft_3], '*', c=colors[1])
    plt.title('After - with segmentation')
    plt.suptitle('R peaks - ' + title)
    plt.savefig('results/Factory/' + 'R peaks-' + title_im)

    rr_bpm_bef = cm.samples_to_bpm(rr_dist_bef)
    rr_bpm_aft_1 = cm.samples_to_bpm(rr_dist_aft_1)
    rr_bpm_aft_3 = cm.samples_to_bpm(rr_dist_aft_3)

    plt.figure()
    ax1 = plt.subplot(311)
    ax1.plot(rr_bpm_bef, c=colors[-1])
    ax1.plot(out_up_bef, rr_bpm_bef[out_up_bef], 'o', c=colors[0])
    ax1.plot(out_low_bef, rr_bpm_bef[out_low_bef], 'o', c=colors[1])
    ax1.axes.get_xaxis().set_visible(False)
    plt.title('Before')
    ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)
    ax2.plot(rr_bpm_aft_1, c=colors[-1])
    ax2.plot(out_up_aft_1, rr_bpm_aft_1[out_up_aft_1], 'o', c=colors[0])
    ax2.plot(out_low_aft_1, rr_bpm_aft_1[out_low_aft_1], 'o', c=colors[1])
    ax2.axes.get_xaxis().set_visible(False)
    plt.title('After - no segmentation')
    ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
    ax3.plot(rr_bpm_aft_3, c=colors[-1])
    ax3.plot(out_up_aft_3, rr_bpm_aft_3[out_up_aft_3], 'o', c=colors[0])
    ax3.plot(out_low_aft_3, rr_bpm_aft_3[out_low_aft_3], 'o', c=colors[1])
    plt.title('After - with segmentation')
    plt.suptitle('BPM - ' + title)
    plt.savefig('results/Factory/' + 'BPM-' + title_im)

    sub_series = pd.Series(subject, name='Subject')
    start = pd.Series(start_min, name='StartMin')
    end = pd.Series(end_min, name='EndMin')
    missing_bef = pd.Series(miss_bef, name='MissPeaksBefore')
    missing_aft1 = pd.Series(miss_aft_1, name='MissPeaksAfter')
    time_1 = pd.Series(time_no_seg, name='Time')
    missing_aft3 = pd.Series(miss_aft_3, name='MissPeaksAfter_seg')
    time_3 = pd.Series(time_seg_align, name='Time_seg')
    wrong_bef = pd.Series(wron_bef, name='WrongPeaksBefore')
    wrong_aft1 = pd.Series(wron_aft_1, name='WrongPeaksAfter')
    wrong_aft3 = pd.Series(wron_aft_3, name='WrongPeaksAfter_seg')

    res_df = pd.concat([sub_series, start, end, missing_bef, wrong_bef, missing_aft1, wrong_aft1, time_1, missing_aft3,
                        wrong_aft3, time_3], axis=1)

    res_df.to_csv('results/Factory/results.txt', mode='a',
                  columns=['Subject', 'StartMin', 'EndMin', 'MissPeaksBefore',
                           'WrongPeaksBefore', 'MissPeaksAfter',
                           'WrongPeaksAfter', 'Time', 'MissPeaksAfter_seg',
                           'WrongPeaksAfter_seg', 'Time_seg'], index=False)

