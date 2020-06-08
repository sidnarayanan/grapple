#!/usr/bin/env python

pattern = '''
dataset_pattern: /data/t3home000/bmaier/data_preconvert_9000/WJets_*npz 
output: /data/t3home000/snarayan/papu/expts/1/NAME/ 
n_epochs: 50

embedding_size: HID
hidden_size: HID
intermediate_size: HID
feature_size: 6 
num_hidden_layers: 1 
num_attention_heads: 4
num_encoders: DEPTH
label_size: 1
attention_band: BAND

batch_size: BATCH
mask_charged: True 
lr: LR
lr_decay: 0.99
num_max_files: 20
num_max_particles: 9000
#min_met: 50
plot: /home/snarayan/public_html/figs/papu/scan1/NAME/
pt_weight: True 
'''

def write(hid, depth, band, batch):
    yaml = pattern[:]
    yaml = yaml.replace('HID', str(hid))
    yaml = yaml.replace('DEPTH', str(depth))
    yaml = yaml.replace('BATCH', str(batch))
    yaml = yaml.replace('BAND', str(band))
    lr = max(0.0005, 0.001 * batch / 64)
    yaml = yaml.replace('LR', str(lr))
    name = '%i_%i_%i_%i'%(hid, depth, band, batch)
    yaml = yaml.replace('NAME', name)
    with open(name+'.yaml', 'w') as fyaml:
        fyaml.write(yaml.strip())


write(8, 4, 20, 64)
write(8, 8, 20, 64)
write(16, 4, 20, 32)
write(16, 4, 20, 64)
write(32, 4, 20, 32)
write(16, 6, 20, 32)
write(16, 6, 20, 8)
