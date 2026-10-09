"""Monitoring must preserve facts and old theories without changing training."""
import json
from pathlib import Path

from owt import research


def test_csv_ignores_partial_rows_and_uses_latest_complete_update(tmp_path):
    path=tmp_path/'metrics.csv'
    path.write_text('optimizer_step,val_nll,elapsed_seconds\n'
                    '500,4.1,100\n500,4.2,101\n1000,4.0,200\n1500,3.')
    rows=research.read_csv(path)
    assert [row['optimizer_step'] for row in rows] == [500,1000]
    assert rows[0]['val_nll'] == 4.2


def test_activity_log_is_deduplicated_and_tex_safe(tmp_path):
    assert research.record_event('x','A 5% change','weights_1 & weights_2',doc=tmp_path)
    original=(tmp_path/'research_activity_log.tex').read_text()
    assert not research.record_event('x','altered','replacement',doc=tmp_path)
    assert (tmp_path/'research_activity_log.tex').read_text() == original
    assert r'5\%' in original and r'weights\_1 \& weights\_2' in original
    assert len((tmp_path/'research_events.jsonl').read_text().splitlines()) == 1


def test_revision_archives_old_theory_and_frozen_evidence(tmp_path):
    (tmp_path/'current_theory.tex').write_text('OLD\n'+r'\input{outputs/research-notes/generated_status.tex}')
    (tmp_path/'generated_status.tex').write_text('FROZEN EVIDENCE')
    (tmp_path/'theory_versions.json').write_text('{"current_version":1,"revisions":[]}')
    incoming=tmp_path/'new.tex'
    incoming.write_text('NEW THEORY')
    research.archive_theory(incoming,'A new result changes the hypothesis.',doc=tmp_path)
    assert (tmp_path/'history/theory-v001.tex').read_text() == 'OLD\nFROZEN EVIDENCE'
    assert (tmp_path/'current_theory.tex').read_text() == 'NEW THEORY'
    assert json.loads((tmp_path/'theory_versions.json').read_text())['current_version'] == 2
    assert 'theory-v001.tex' in (tmp_path/'theory_history.tex').read_text()


def test_observation_reports_matched_update_not_latest_baseline(tmp_path):
    for name,rows in [('mdm','500,7.0,100\n5000,4.0,1000\n'),
                      ('mdm_np_zero_init','500,6.5,150\n')]:
        folder=tmp_path/name/'local_metrics'
        folder.mkdir(parents=True)
        (folder/'validation.csv').write_text('optimizer_step,val_nll,elapsed_seconds\n'+rows)
    snapshot=research.collect(tmp_path)
    assert snapshot['variants']['mdm_np_zero_init']['matched_validation_delta'] == -.5
    tex=research.status_tex(snapshot)
    assert '500 & 7.0000 & -- & 6.5000 & --' in tex


def test_theory_archive_freezes_figure_bytes(tmp_path):
    figure = tmp_path/'figure.pdf'
    figure.write_bytes(b'original figure bytes')
    (tmp_path/'current_theory.tex').write_text(r'\includegraphics[width=\linewidth]{'+str(figure)+'}')
    incoming = tmp_path/'new.tex'
    incoming.write_text('new theory')
    research.archive_theory(incoming, 'New evidence.', doc=tmp_path)
    figure.write_bytes(b'changed figure')
    frozen = list((tmp_path/'history/theory-v000-assets').iterdir())
    assert len(frozen) == 1 and frozen[0].read_bytes() == b'original figure bytes'
    archived = (tmp_path/'history/theory-v000.tex').read_text()
    assert str(frozen[0]) in archived and str(figure) not in archived
