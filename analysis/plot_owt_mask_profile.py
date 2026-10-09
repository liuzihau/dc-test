"""Publication-style plots of the frozen exploratory checkpoint diagnostic."""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/runtime/analysis/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'outputs/analysis/owt-mask-profile-5000/summary.json')
    args=parser.parse_args()
    data=json.loads(args.input.read_text())
    label={'mdm_np':'Random NP','mdm_np_zero_init':'Zero-init NP',
           'mdm_np_zero_init_low_weight':'Low-weight NP'}[data.get('comparison_variant','mdm_np')]
    rows=data['summary']
    t=np.array([r['noise_level'] for r in rows])
    delta=np.array([r['delta_main_elbo'] for r in rows])
    interval=np.array([r['diagnostic_row_bootstrap_95'] for r in rows])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5))
    axes[0].errorbar(t,delta,yerr=np.maximum(0,np.vstack([delta-interval[:,0],interval[:,1]-delta])),
                     fmt='o-',color='#2166ac',capsize=4,lw=1.8)
    axes[0].axhline(0,color='#777777',ls='--',lw=1)
    axes[0].set(title=f'Main prediction: {label} minus MDM',
                ylabel='Difference in main ELBO (nats/token)')
    for key,label,color in [('paired_visible_source_ce','Visible source','#238b45'),
                            ('paired_masked_source_ce','Masked source','#d95f0e'),
                            ('paired_self_ce','Own target position','#756bb1')]:
        axes[1].plot(t,[r[key] for r in rows],marker='o',label=label,color=color,lw=1.8)
    axes[1].set(title='Existing NP readouts on identical targets',ylabel='Cross-entropy (nats/masked target)')
    axes[1].legend(frameon=False,fontsize=9)
    for ax in axes:
        ax.set_xlabel('Corruption level t')
        ax.set_xticks(t)
        ax.grid(alpha=.2)
    fig.suptitle('Frozen final EMA checkpoints: exploratory CPU FP32 profile',fontsize=12)
    fig.text(.5,.01,f'{len(data["row_ids"])} fixed held-out rows; paired corruption. '
             'Error bars resample rows, not training seeds. Readouts do not prove downstream use.',
             ha='center',fontsize=8.4,color='#555555')
    fig.tight_layout(rect=(0,.055,1,.95))
    for extension in ['pdf','png']:
        fig.savefig(args.input.parent/f'mask_profile.{extension}',dpi=180)
    plt.close(fig)
    print(args.input.parent/'mask_profile.pdf')


if __name__=='__main__':main()
