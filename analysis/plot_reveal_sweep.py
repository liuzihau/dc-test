"""Heatmaps of conditional masked-target performance and reveal sensitivity."""
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
from owt.research import LABELS
from owt.reveal_sweep import MASK_RATIOS,CORRECT_FRACTIONS,COSINE_GROUPS
from matplotlib.backends.backend_pdf import PdfPages


def panel(ax,values,title,cmap,vmin,vmax,percent=False,signed=True):
    picture=ax.imshow(values,cmap=cmap,vmin=vmin,vmax=vmax,aspect='auto')
    ax.set(title=title,xlabel='Correct revealed tokens (%)',ylabel='Masked input (%)',
           xticks=range(3),xticklabels=[100,80,60],yticks=range(5),yticklabels=[100,80,60,40,20])
    for i in range(5):
        for j in range(3):
            fraction=(values[i,j]-vmin)/max(vmax-vmin,1e-12)
            color='white' if cmap=='viridis' and fraction<.45 else 'black'
            label=f'{values[i,j]:.1f}' if percent else (f'{values[i,j]:+.3f}' if signed else f'{values[i,j]:.3f}')
            ax.text(j,i,label,
                    ha='center',va='center',fontsize=8.5,color=color)
    return picture


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,help='Fresh figure directory; defaults to the input directory')
    parser.add_argument('--skip-cosines',action='store_true',help='Render only the performance and difference heatmaps')
    args=parser.parse_args()
    output=args.output or args.input.parent
    output.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.input.read_text());rows=data['summary']
    variants=data['protocol']['variants']
    lookup={(r['variant'],r['mask_ratio'],r['correct_fraction']):r for r in rows}
    def matrix(variant,key,scale=1.):
        return np.array([[lookup[(variant,m,c)][key]*scale for c in CORRECT_FRACTIONS] for m in MASK_RATIOS])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,len(variants),figsize=(3.7*len(variants),7.7),squeeze=False)
    for k,(metric,label,scale) in enumerate([('masked_ce','Masked-target CE: lower is better',1.),
                                            ('masked_accuracy','Masked-target accuracy (%): higher is better',100.)]):
        matrices=[matrix(v,metric,scale) for v in variants]
        lo=min(x.min() for x in matrices);hi=max(x.max() for x in matrices)
        for j,(v,values) in enumerate(zip(variants,matrices)):
            picture=panel(axes[k,j],values,LABELS[v]+'\n'+label,'viridis',lo,hi,metric=='masked_accuracy',signed=False)
            fig.colorbar(picture,ax=axes[k,j],fraction=.045,pad=.025)
    n=len(data['protocol']['row_ids'])
    footer=(f'{n} fixed held-out rows; final EMA; one training seed.\n'
            'Identical inputs across models; conditional diagnostic, not generation quality.\n'
            'Wrong reveals copy the nearest different clean token; masked targets may be exposed.')
    fig.suptitle('Masking × revealed-token reliability',fontsize=13)
    fig.text(.5,.012,footer,ha='center',fontsize=8,color='#555555')
    fig.tight_layout(rect=(0,.10,1,.95))
    for extension in ['pdf','png']:
        fig.savefig(output/f'reveal_performance.{extension}',dpi=180)
    plt.close(fig)
    others=[v for v in variants if v!='mdm']
    if others:
        fig,axes=plt.subplots(2,len(others),figsize=(4.4*len(others),7.7),squeeze=False)
        for k,(metric,label) in enumerate([('delta_from_mdm','Main CE minus MDM'),
                         ('excess_reliability_penalty_vs_mdm','Extra wrong-reveal penalty versus MDM')]):
            matrices=[matrix(v,metric) for v in others]
            bound=max(.001,max(float(np.abs(x).max()) for x in matrices))
            for j,(v,values) in enumerate(zip(others,matrices)):
                picture=panel(axes[k,j],values,LABELS[v]+'\n'+label,'RdBu_r',-bound,bound)
                fig.colorbar(picture,ax=axes[k,j],fraction=.045,pad=.025)
        fig.suptitle('Main performance and\nwrong-reveal sensitivity',fontsize=12)
        fig.text(.5,.015,f'{n} paired rows; positive values are worse for NP.\n'
                 'Reliability penalty subtracts each model’s\n'
                 '100%-correct condition. Row intervals are in summary.json.',
                 ha='center',fontsize=8,color='#555555')
        fig.tight_layout(rect=(0,.10,1,.93))
        for extension in ['pdf','png']:
            fig.savefig(output/f'reveal_differences.{extension}',dpi=180)
        plt.close(fig)
    if data.get('layerwise_cosines') and not args.skip_cosines:
        cosine={(r['variant'],r['mask_ratio'],r['correct_fraction'],r['token_group'],r['layer']):r
                for r in data['layerwise_cosines']}
        colors=['#555555','#2166ac','#238b45','#d95f0e']
        titles=['Masked positions','Correct revealed positions','Incorrect revealed positions']
        with PdfPages(output/'layer_cosine_sweep.pdf') as pages:
            for m in MASK_RATIOS:
                for c in CORRECT_FRACTIONS:
                    fig,axes=plt.subplots(1,3,figsize=(11.5,4.4))
                    for ax,group,title in zip(axes,COSINE_GROUPS,titles):
                        drawn=False
                        for variant,color in zip(variants,colors):
                            points=[cosine[(variant,m,c,group,l)] for l in range(1,13)]
                            if points[0]['mean_cosine'] is None:
                                continue
                            means=[p['mean_cosine'] for p in points]
                            low=[p['row_bootstrap_95'][0] for p in points]
                            high=[p['row_bootstrap_95'][1] for p in points]
                            ax.plot(range(1,13),means,'o-',color=color,label=LABELS[variant],lw=1.5,ms=3)
                            ax.fill_between(range(1,13),low,high,color=color,alpha=.13)
                            drawn=True
                        if not drawn:
                            ax.text(.5,.5,'No eligible positions',ha='center',va='center',transform=ax.transAxes)
                        ax.set(title=title,xlabel='Transformer block',ylabel='Before/after block cosine',
                               xticks=range(1,13,2))
                        ax.grid(alpha=.2)
                        if drawn:
                            ax.legend(frameon=False,fontsize=8)
                    fig.suptitle(f'Masked input {m:.0%}; correct revealed tokens {c:.0%}',fontsize=13)
                    fig.text(.5,.015,f'{n} paired rows; cosine excludes position zero and clean EOS. '
                             'Bands resample rows, not training seeds.\n'
                             'Mean within each row, then average rows. Geometric change does not establish forgetting or useful computation.',
                             ha='center',fontsize=8,color='#555555')
                    fig.tight_layout(rect=(0,.08,1,.94))
                    name=f'layer_cosine_mask{round(m*100):03d}_correct{round(c*100):03d}'
                    for extension in ['pdf','png']:
                        fig.savefig(output/f'{name}.{extension}',dpi=180)
                    pages.savefig(fig)
                    plt.close(fig)
    print(output/'reveal_performance.pdf')


if __name__=='__main__':
    main()
