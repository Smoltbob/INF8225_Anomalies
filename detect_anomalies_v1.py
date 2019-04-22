"""
This script tests the trained model by applying it to detect abnormal events in the specified sequences.
"""
import numpy as np
import os
import utilities as util
import networks as net
import matplotlib.pyplot as plt

# Get all the training data
motion_data = util.ExtractedData()
video_names = [v for v in os.listdir(util.test_dir) if os.path.isdir(os.path.join(util.test_dir, v))
               and util.test_dir_name.lower() in v.lower()]
video_names.sort()
for video_name in video_names:
    motion_data.add_data(video_name, open_if_exist=True)

# Files setup
method_name = 'cae_method_ped1_v3'
result_dir = os.path.join(util.root_dir_name, '{}/{}'.format(util.result_dir_name, method_name))
models_dir = os.path.join(util.root_dir_name, '{}/{}'.format(util.model_dir_name, method_name))

# Ref.: https://stackoverflow.com/questions/29451030/why-doesnt-np-genfromtxt-remove-header-while-importing-in-python
with open(os.path.join(result_dir, util.summary_results_filename), 'r') as results:
    line = results.readline()
    header = [e for e in line.strip().split(',') if e]
    results_array = np.genfromtxt(results, names=header, dtype=None, delimiter=',')

auc_results = []

for i in range(net.repeat_number):
    print('======================== Iteration {} ========================'.format(i))
    # Get the ground-truth anomaly labels
    gt_labels = util.get_gt_results()

    scores = []
    for v, sequence_name in enumerate(video_names):
        print('Applying trained models to Video {}:'.format(sequence_name))
        # Get the test data
        img_t, pre_t, nex_t = motion_data.get_testing_data(sequence_name)
        # Apply trained models
        _, test_scores = net.test_trained_ae_model(test_roi_image_data=img_t,
                                                   test_prev_diff_data=pre_t,
                                                   test_next_diff_data=nex_t,
                                                   model_dir_path=models_dir,
                                                   iteration_number=i)
        # Get smoothed results
        f_t, s_t = motion_data.get_results(sequence_name, test_scores)
        scores.append(s_t.tolist())
        # Plot a curve of scores and threshold_value
        assert len(gt_labels[v]) == len(s_t), 'Something is wrong with the length of result sequence array.'
        gt_scores = np.array(gt_labels[v])
        gt_scores = np.interp(gt_scores, (gt_scores.min(), gt_scores.max()), (s_t.min(), s_t.max()))
        figure_name = 'anomaly_curve_in_video_{}_model_{}.pdf'.format(sequence_name, i)
        fig = plt.figure(figsize=(10, 4))
        plt.plot(f_t, s_t, 'r', label='score', lw=3)
        plt.fill_between(f_t, s_t.min(), gt_scores, facecolor='c', label='ground-truth', lw=1)
        plt.xlabel('Frame')
        plt.ylabel('Reconstruction error')
        plt.legend(loc='best')
        fig.savefig(os.path.join(result_dir, figure_name), bbox_inches='tight')
        print('==============================================================')

    print('Computing overall performance:')
    # Compute the AUC
    auc_result = net.get_global_result(gt_results=gt_labels,
                                       ts_results=scores,
                                       result_dir_path=result_dir,
                                       iteration_number=i)

    output_string = '\nAUC = {:.2f}% using {}.'.format(auc_result*100.0, method_name)
    print(output_string)
    auc_results.append(auc_result)

    if i == 0:
        with open(os.path.join(result_dir, util.detect_results_filename), 'wb') as summary_file:
            summary_file.write(b'iteration,AUC\n')

    with open(os.path.join(result_dir, util.detect_results_filename), 'ab') as summary_file:
        np.savetxt(summary_file, np.array([i, auc_result]).reshape(1, -1), delimiter=',')

    print('==============================================================')

auc_results = np.array(auc_results)

# The best model is the one that as the maximum sum of ratios
best_index = np.argmax(auc_results)

best_trained_model_iteration = int(results_array['iteration'][best_index])
best_auc_result = auc_results[best_index]

# Detect summary
detect_summary_file = open(os.path.join(result_dir, util.global_detect_summary_filename), 'w')

output_string = 'Detect summary of frame-level anomaly detection with {}\n'.format(method_name)
output_string += '---------------------------------------------------------------------\n'

output_string += '\nThe best one is model_{} with {:.2f}% of AUC.\n'.format(best_trained_model_iteration,
                                                                            best_auc_result*100.0)

detect_summary_file.write(output_string)
print(output_string)

detect_summary_file.close()
