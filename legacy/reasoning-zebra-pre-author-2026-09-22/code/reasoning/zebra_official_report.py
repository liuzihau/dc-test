"""Size-stratified generated accuracy on prepared source-aligned Zebra data."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .data import ReasoningDataset
from .tasks import _strip_answer, score_prediction
from .zebra_official import parse_prompt


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def wilson(correct, total):
    if not total:
        return None, None
    p, z = correct / total, 1.959963984540054
    denominator = 1 + z*z / total
    center = (p + z*z / (2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return max(0., center-half), min(1., center+half)


def _finish(houses, attributes, counts):
    low, high = wilson(counts['solved'], counts['examples'])
    return dict(houses=houses, attributes=attributes, **counts,
                solve_accuracy=counts['solved']/counts['examples'] if counts['examples'] else None,
                solve_ci95_low=low, solve_ci95_high=high,
                content_accuracy=counts['content_correct']/counts['content_tokens'] if counts['content_tokens'] else None)


def summarize(evaluation, data_dir, output_dir):
    evaluation, data_dir, output = Path(evaluation).resolve(), Path(data_dir).resolve(), Path(output_dir).resolve()
    if output == data_dir or data_dir in output.parents:
        raise ValueError('Write reports outside immutable prepared data')
    if evaluation.parent == output and evaluation.name in ('accuracy_by_size.csv', 'official_zebra_accuracy_by_size.png', 'summary.json', 'summary.md'):
        raise ValueError('Report output would overwrite input evaluation')
    data = json.loads(evaluation.read_text())
    args, metrics, contract = data['arguments'], data['metrics'], data['contract']
    if (contract['task'] != 'zebra-official' or args['protocol'] != 'generate'
            or args['split'] not in ('test', 'validation') or args['policy'] != 'top_prob'
            or args['memory_condition'] != 'correct' or metrics['evaluation'] != 'closed_loop_generation'
            or metrics['policy'] != 'top_prob' or metrics['candidate_k'] != 8
            or metrics['memory_condition'] != 'correct' or metrics['token_selection'] != 'paper'
            or metrics['tokens_per_step'] != 1 or metrics['max_steps'] is not None
            or metrics['seed'] != args['seed'] or metrics['mean_nfe_per_example'] != 37):
        raise ValueError('Requires complete standard generated evaluation on zebra-official')
    if sha256(data_dir / 'manifest.json') != contract['data_sha256']:
        raise ValueError('Prepared manifest differs from checkpoint/evaluation data contract')
    dataset = ReasoningDataset(data_dir, args['split'])
    examples = data['examples']
    if (not examples or len(examples) != metrics['num_examples']
            or len(examples) != min(args['examples'], len(dataset))):
        raise ValueError('Evaluation example count differs from protocol')
    # The runner evaluates the immutable prefix of its selected split.
    if [item['record_index'] for item in examples] != list(range(len(examples))):
        raise ValueError('Evaluation record indices are not the declared split prefix')
    counts = {(h, a): dict(examples=0, solved=0, exact_matches=0, wellformed=0,
                           content_correct=0, content_tokens=0)
              for h in range(3, 7) for a in range(3, 7)}
    for example in examples:
        record = dataset.records[example['record_index']]
        if (example['id'] != record['id'] or example['task'] != 'zebra-official'
                or not example['all_slots_completed'] or example['remaining_masked_slots'] != 0
                or example['token_selection'] != 'sample' or example['nfe'] != 37):
            raise ValueError('Evaluation IDs/task/completion do not match prepared records')
        houses, attributes, _ = parse_prompt(record['prompt'])
        if (record['metadata']['houses'], record['metadata']['attributes']) != (houses, attributes):
            raise ValueError('Puzzle-size metadata disagrees with public prompt')
        if not score_prediction(record, record['answer'])['valid_solution']:
            raise ValueError('Prepared ground truth fails actual task scorer')
        raw = example['predicted_answer_slots']
        if len(raw) != 37 or score_prediction(record, raw) != example['scores']:
            raise ValueError('Saved prediction scores differ from actual task scorer or fixed layout')
        answer = _strip_answer(raw)
        size = houses * attributes
        formed = answer is not None and len(answer) == size and all(token in tuple(map(str, range(houses))) for token in answer)
        group = counts[houses, attributes]
        group['examples'] += 1
        group['solved'] += int(example['scores']['valid_solution'])
        group['exact_matches'] += int(example['scores']['exact_match'])
        group['wellformed'] += int(formed)
        group['content_tokens'] += size
        group['content_correct'] += sum(a == b for a, b in zip(answer, record['answer'])) if formed else 0
    totals = {key: sum(group[key] for group in counts.values()) for key in next(iter(counts.values()))}
    overall = _finish('all', 'all', totals)
    if (not math.isclose(overall['solve_accuracy'], metrics['valid_solution'], abs_tol=1e-12)
            or not math.isclose(totals['exact_matches']/totals['examples'], metrics['exact_match'], abs_tol=1e-12)):
        raise ValueError('Rescored aggregates differ from evaluation metrics')
    provenance = dict(evaluation=str(evaluation), evaluation_sha256=sha256(evaluation),
                      checkpoint=data['checkpoint'], step=data['step'], dataset_sha256=contract['data_sha256'],
                      split=args['split'], split_sha256=dataset.manifest['splits'][args['split']]['sha256'],
                      checkpoint_verification='Recorded evaluation metadata; checkpoint not available at recorded path')
    checkpoint = Path(data['checkpoint'])
    if checkpoint.is_file():
        checkpoint = checkpoint.resolve()
        receipt = json.loads(checkpoint.with_suffix('.pt.json').read_text())
        if (receipt['step'] != data['step'] or receipt['size'] != checkpoint.stat().st_size
                or receipt['sha256'] != sha256(checkpoint)):
            raise ValueError('Checkpoint receipt integrity/step mismatch')
        provenance.update(checkpoint_sha256=receipt['sha256'], checkpoint_verification='Adjacent receipt/size/SHA256 verified; no pickle loading')
    rows = [_finish(h, a, group) for (h, a), group in counts.items()]
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / 'accuracy_by_size.csv'
    if csv_path == evaluation:
        raise ValueError('Output collides with input evaluation')
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows + [overall])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    matrix = np.array([[100*counts[h, a]['solved']/counts[h, a]['examples'] if counts[h, a]['examples'] else np.nan
                        for a in range(3, 7)] for h in range(3, 7)])
    fig, axis = plt.subplots(figsize=(8.4, 7.2))
    image = axis.imshow(matrix, vmin=0, vmax=100, cmap='Blues')
    for row in rows:
        text = ('No examples' if not row['examples'] else
                f"{100*row['solve_accuracy']:.1f}% ({row['solved']}/{row['examples']})\n95% CI [{100*row['solve_ci95_low']:.1f}, {100*row['solve_ci95_high']:.1f}]")
        axis.text(row['attributes']-3, row['houses']-3, text, ha='center', va='center', fontsize=8,
                  color='white' if (row['solve_accuracy'] or 0) > .6 else 'black')
    axis.set(xticks=range(4), xticklabels=range(3, 7), yticks=range(4), yticklabels=range(3, 7),
             xlabel='Attributes', ylabel='Houses', title=f"Source-aligned Zebra · {contract['variant']} · step {data['step']}\nOverall exact solving: {100*overall['solve_accuracy']:.2f}% ({totals['solved']}/{totals['examples']})")
    fig.colorbar(image, ax=axis, label='Whole-puzzle exact solving (%)')
    fig.text(.5, .015, 'Same source family, not the paper’s complete training/evaluation reproduction. Wilson intervals describe puzzle sampling, not training-seed variation.', ha='center', fontsize=7)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(output / 'official_zebra_accuracy_by_size.png', dpi=180)
    plt.close(fig)
    report = dict(overall=overall, by_size=rows, provenance=provenance)
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    low, high = overall['solve_ci95_low'], overall['solve_ci95_high']
    lines = ['# Source-aligned Zebra reference', '',
             f"Model: `{contract['variant']}`; completed updates: {data['step']}; split: `{args['split']}`.", '',
             f"Whole-puzzle exact solving: **{100*overall['solve_accuracy']:.2f}%** ({totals['solved']}/{totals['examples']}); Wilson 95% interval [{100*low:.2f}%, {100*high:.2f}%].",
             f"Content-token accuracy: **{100*overall['content_accuracy']:.2f}%** ({totals['content_correct']}/{totals['content_tokens']}). EOS and PAD are excluded; malformed answers count wrong on all houses × attributes content positions.", '',
             'See `accuracy_by_size.csv` and `official_zebra_accuracy_by_size.png` for all 16 puzzle sizes and their denominators.', '',
             'This is source-data alignment, **not an exact reproduction of the paper’s 96.9% result**. Training amount, model details, answer serialization and evaluation subset can differ. Do not compare this score directly with our separate synthetic 5×5 pilot.',
             'The model uses 37 fixed answer slots (content + EOS + PAD). Predicting easy PAD/EOS tokens can lower training/validation loss without improving reasoning; the content metric and exact solving score above exclude that shortcut.',
             'Wilson intervals measure finite-puzzle uncertainty only, not uncertainty across training seeds. No general reasoning-improvement claim follows from one reference run.']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation', 'data-dir', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    result = summarize(args.evaluation, args.data_dir, args.output_dir)
    print(json.dumps(result['overall']))


if __name__ == '__main__':
    main()
