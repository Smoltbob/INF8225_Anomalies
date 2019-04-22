"""
This script trains model for anomaly detecting in videos.
"""
import numpy as np
import os
import utilities as util
import networks as net
from natsort import natsorted

# Get all the training data
motion_data = util.ExtractedData()
video_names = [v for v in os.listdir(util.train_dir) if os.path.isdir(os.path.join(util.train_dir, v))
               and util.train_dir_name.lower() in v.lower()]
video_names = natsorted(video_names)
for video_name in video_names:
    motion_data.add_data(video_name, open_if_exist=True)

best_layer_type = (256, 128, 64)

# Files setup
method_name = 'cae_method_ped2_v3'
result_dir = os.path.join(util.root_dir_name, '{}/{}'.format(util.result_dir_name, method_name))
models_dir = os.path.join(util.root_dir_name, '{}/{}'.format(util.model_dir_name, method_name))

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

normal_train_ratio_list = []
normal_valid_ratio_list = []

for i in range(net.repeat_number):
    print('======================== Iteration {} ========================'.format(i))
    # Shuffle the data by row only
    # and get the seed in order to reproduce the random sequence
    img_t, pre_t, nex_t, img_v, pre_v, nex_v, rand_seed = motion_data.get_training_validation_data()

    mv1 = net.BuildOurMethodV1(hidden_units=best_layer_type, model_dir_path=models_dir, iteration_number=i)

    mv1.train(train_roi_image_data=img_t, train_prev_diff_data=pre_t, train_next_diff_data=nex_t,
              train_cfv_ae_only=False)

    # Get the scores
    ae_train_score = mv1.global_mse
    ae_train_scores = mv1.mse_per_sample

    ae_validate_score, ae_validate_scores = net.test_trained_ae_model(test_roi_image_data=img_v,
                                                                      test_prev_diff_data=pre_v,
                                                                      test_next_diff_data=nex_v,
                                                                      model_dir_path=models_dir,
                                                                      iteration_number=i)

    output_string = 'Iteration {} with layer type {}: ae_train_score = {}; ae_validation_score = {}'\
        .format(i, best_layer_type, ae_train_score, ae_validate_score)

    print('\n')

    # Save the result to a global summary file
    output_string += '\n'

    threshold_value = ae_train_score + ae_validate_score + 3 * (np.std(ae_train_scores) + np.std(ae_validate_scores))

    # Summary file format: [Iteration, ae_train_score, ae_validate_score, threshold_value,
    #                       normal_train_ratio, normal_valid_ratio, abnormal_ratio]
    if i == 0:
        with open(os.path.join(result_dir, util.summary_results_filename), 'wb') as summary_file:
            summary_file.write(b'iteration,random_shuffle_seed,ae_train_score,ae_validate_score,threshold_value,'
                               b'normal_train_ratio,normal_valid_ratio\n')

    normal_train_ratio = sum([score < threshold_value for score in ae_train_scores])/float(len(ae_train_scores))
    normal_valid_ratio = sum([score < threshold_value for score in ae_validate_scores])/float(len(ae_validate_scores))

    normal_train_ratio_list.append(normal_train_ratio*100.0)
    normal_valid_ratio_list.append(normal_valid_ratio*100.0)

    with open(os.path.join(result_dir, util.summary_results_filename), 'ab') as summary_file:
        np.savetxt(summary_file, np.array([i, rand_seed, ae_train_score, ae_validate_score, threshold_value,
                                           normal_train_ratio, normal_valid_ratio]).reshape(1, -1),
                   delimiter=',')

    output_string += '{0:.2f}% of normal training samples are detected as normal.\n'.format(normal_train_ratio*100.0)
    output_string += '{0:.2f}% of normal validation samples are detected as normal.\n'.format(normal_valid_ratio*100.0)

    print(output_string)
    print('==============================================================')

# Global summary
global_summary_file = open(os.path.join(result_dir, util.global_summary_filename), 'w')

output_string = 'Global summary of abnormal event detection with {}\n'.format(method_name)
output_string += '---------------------------------------------------------------------\n'
output_string += 'On average, using layer type {},\n'.format(best_layer_type)
output_string += '\t{0:.2f}% of normal training samples are detected as normal;\n'.format(np.mean(
    normal_train_ratio_list))
output_string += '\t{0:.2f}% of normal validation samples are detected as normal;\n'.format(np.mean(
    normal_valid_ratio_list))

global_summary_file.write(output_string)
print(output_string)

global_summary_file.close()
