from src.wsd.utils.utils import yield_batch


def get_instances(path):

    instances = {}
    for line in open(path):
        instance, answ = line.strip().split(' :: ')
        instances[instance] = answ

    return instances

def get_instances_from_hr(path):
    instances = {}

    for batch in yield_batch(path, separator='\n'):
        instances_info = batch[0]
        input_sent = batch[1]
        gold_sent = batch[2]
        generated = batch[3]
        clean = batch[4]
        instance_key = ' '.join(instances_info.split()[:-1])
        instances[instance_key] = {'generated': generated, 'clean':clean}
    return instances

if __name__ == '__main__':
    path_a = 'checkpoints/bart_313_pt_semcor_0.7_drop_0.1_enc_lyd_0.6_dec_lyd_0.2/beams_50_return_50/output_files/coinco_twsi_cut_per_target_csi_extended_vocab_best.txt'
    path_b = 'checkpoints/bart_313_pt_semcor_0.7_drop_0.1_enc_lyd_0.6_dec_lyd_0.2/beams_50_return_50/output_files/coinco_twsi_cut_per_target_output_analysis_best.txt'

    hr_a = 'checkpoints/bart_313_pt_semcor_0.7_drop_0.1_enc_lyd_0.6_dec_lyd_0.2/beams_50_return_50/output_files/coinco_twsi_cut_per_target_hr_csi_extended_vocab_output.txt'
    hr_b = 'checkpoints/bart_313_pt_semcor_0.7_drop_0.1_enc_lyd_0.6_dec_lyd_0.2/beams_50_return_50/output_files/coinco_twsi_cut_per_target_hr_output_analysis_output.txt'

    readable_a = get_instances_from_hr(hr_a)
    readable_b = get_instances_from_hr(hr_b)

    instances_a = get_instances(path_a)
    instances_b = get_instances(path_b)

    for inst in instances_a:
        if inst not in instances_b and inst.split()[0].split('.')[-1]!='r':
            print(inst)
            if inst in readable_a and inst in readable_b:
                print(readable_a[inst], readable_b[inst])