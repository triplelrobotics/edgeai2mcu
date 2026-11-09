import csv
import io
import subprocess
import json
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_dataset as prepare
from prepare_dataset import ReviewStore


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sources = []
        for n, name in enumerate(('session_A', 'session_B', 'session_C')):
            source = self.root / name
            (source / 'images').mkdir(parents=True)
            rows = []
            for i, label in enumerate(['STOP', 'RIGHT', 'STRAIGHT', 'LEFT']):
                filename = f'{i+1:06d}.jpg'
                pixels = np.full((16, 16, 3), 20 + n * 60 + i * 10, np.uint8)
                cv2.imwrite(str(source / 'images' / filename), pixels)
                rows.append([filename, label, str(100 + i), str(99 + i)])
            with (source / 'labels.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(prepare.SOURCE_FIELDS[:4])
                writer.writerows(rows)
            self.sources.append({'path': str(source), 'review': str(source / 'review.csv')})
        self.paths = [Path(item['path']) for item in self.sources]

    def approve(self):
        for source in self.sources:
            store = ReviewStore(source['path'], source['review'])
            store.update({'revision': 0, 'indices': [1, 2, 3], 'changes': {'decision': 'keep'}})

    def test_check_command_reports_without_writing_files(self):
        source = Path(self.sources[0]['path'])
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        output = io.StringIO()
        with redirect_stdout(output):
            prepare.main(['check', '--source', str(source)])
        self.assertEqual(json.loads(output.getvalue())['samples'], 4)
        after = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_review_command_starts_local_server_and_closes_on_interrupt(self):
        item = self.sources[0]
        with patch('http.server.HTTPServer') as server, redirect_stdout(io.StringIO()):
            server.return_value.server_port = 9876
            server.return_value.serve_forever.side_effect = KeyboardInterrupt
            prepare.main(['review', '--source', item['path'], '--port', '9876'])
        self.assertEqual(server.call_args.args[0], ('127.0.0.1', 9876))
        server.return_value.server_close.assert_called_once()
        self.assertTrue(Path(item['review']).is_file())

    def test_default_review_file_is_inside_raw_session(self):
        source = self.paths[0]
        store = ReviewStore(source)
        self.assertEqual(store.review, source.resolve() / 'review.csv')
        self.assertTrue(store.review.is_file())
        with self.assertRaisesRegex(ValueError, 'Only review.csv'):
            ReviewStore(source, source / 'labels.csv')

    def test_build_command_generates_dataset(self):
        self.approve()
        output = self.root / 'cli_build'
        with redirect_stdout(io.StringIO()):
            prepare.main(['build', '--source', *map(str, self.paths), '--output', str(output)])
        self.assertEqual(prepare.read_classes(output), list(prepare.CLASSES))
        self.assertEqual(len(list(output.rglob('*.jpg'))), 9)

    def test_embedded_page_preserves_ui_and_injects_request_token(self):
        page = prepare.review_page('test-review-token').decode('utf-8')
        self.assertTrue(page.startswith('<!doctype html>'))
        self.assertIn("const token='test-review-token'", page)
        self.assertNotIn('__TOKEN__', page)
        self.assertIn('/api/review', page)
        self.assertIn('Select all filtered', page)

    def test_import_does_not_load_web_server(self):
        script = (f'import sys; sys.path.insert(0, {str(Path(prepare.__file__).parent)!r}); '
                  'import prepare_dataset; assert "http.server" not in sys.modules')
        subprocess.run([sys.executable, '-c', script], check=True, capture_output=True)

    def test_batch_review_undo_resume_and_raw_unchanged(self):
        item = self.sources[0]
        raw = Path(item['path']) / 'labels.csv'
        original = raw.read_bytes()
        store = ReviewStore(item['path'], item['review'])
        store.update({'revision': 0, 'indices': [1, 2], 'changes': {'decision': 'keep', 'reason': 'clear'}})
        resumed = ReviewStore(item['path'], item['review'])
        self.assertEqual(resumed.rows[1]['decision'], 'keep')
        store.update({'revision': 1, 'undo': True})
        self.assertEqual(store.rows[1]['decision'], 'unreviewed')
        self.assertEqual(raw.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, 'another tab'):
            store.update({'revision': 0, 'indices': [0], 'changes': {'decision': 'exclude'}})

    def test_kept_class_counts_follow_decisions_and_corrected_labels(self):
        store = ReviewStore(self.paths[0])
        self.assertEqual(store.payload()['summary']['kept_labels'], {'LEFT': 0, 'RIGHT': 0, 'STRAIGHT': 0})
        # STOP remains outside the driving-class counters even if marked keep.
        store.update({'revision': 0, 'indices': [0, 1, 2], 'changes': {'decision': 'keep'}})
        self.assertEqual(store.payload()['summary']['kept_labels'], {'LEFT': 0, 'RIGHT': 1, 'STRAIGHT': 1})
        store.update({'revision': 1, 'indices': [1], 'changes': {'label_override': 'STRAIGHT'}})
        self.assertEqual(store.payload()['summary']['kept_labels'], {'LEFT': 0, 'RIGHT': 0, 'STRAIGHT': 2})
        store.update({'revision': 2, 'indices': [2], 'changes': {'decision': 'exclude'}})
        self.assertEqual(store.payload()['summary']['kept_labels']['STRAIGHT'], 1)
        store.update({'revision': 3, 'undo': True})
        self.assertEqual(store.payload()['summary']['kept_labels']['STRAIGHT'], 2)

    def test_stale_source_is_rejected(self):
        item = self.sources[0]
        ReviewStore(item['path'], item['review'])
        (Path(item['path']) / 'images/000002.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            ReviewStore(item['path'], item['review'])

    def test_missing_images_require_exclusion_and_cannot_be_kept(self):
        item = self.sources[0]
        (Path(item['path']) / 'images/000002.jpg').unlink()
        store = ReviewStore(item['path'], item['review'])
        self.assertIn('missing_image', store.rows[1]['issues'])
        with self.assertRaisesRegex(ValueError, 'Exclude'):
            store.update({'revision': 0, 'indices': [1], 'changes': {'decision': 'keep'}})
        store.update({'revision': 0, 'indices': [1], 'changes': {'decision': 'exclude'}})

    def test_reproducible_three_class_build_excludes_stop(self):
        self.approve()
        one, two = self.root / 'one', self.root / 'two'
        stats = prepare.build_dataset(self.paths, one)
        prepare.build_dataset(list(reversed(self.paths)), two)
        a = {p.relative_to(one): p.read_bytes() for p in one.rglob('*') if p.is_file()}
        b = {p.relative_to(two): p.read_bytes() for p in two.rglob('*') if p.is_file()}
        self.assertEqual(a, b)
        self.assertEqual(stats['audit']['unreviewed_skipped'], 3)
        self.assertEqual(prepare.read_classes(one), ['LEFT', 'RIGHT', 'STRAIGHT'])
        self.assertEqual(len(list(one.rglob('*.jpg'))), 9)
        self.assertTrue((one / 'train/LEFT').exists())
        with self.assertRaises(FileExistsError):
            prepare.build_dataset(self.paths, one)

    def test_no_kept_samples_blocks_build_without_output(self):
        for item in self.sources:
            ReviewStore(item['path'], item['review'])
        out = self.root / 'out'
        with self.assertRaisesRegex(ValueError, 'No kept driving'):
            prepare.build_dataset(self.paths, out)
        self.assertFalse(out.exists())

    def test_build_uses_only_keep_without_changing_unreviewed_decisions(self):
        self.approve()
        store = ReviewStore(self.paths[0])
        store.update({'revision': 0, 'indices': [1], 'changes': {'decision': 'unreviewed'}})
        before = {p: p.read_bytes() for source in self.paths for p in source.rglob('*') if p.is_file()}
        output = self.root / 'keep_only'
        report = prepare.build_dataset(self.paths, output)
        self.assertEqual(report['audit']['unreviewed_skipped'], 4)
        self.assertEqual(len(list(output.rglob('*.jpg'))), 8)
        with (output / 'manifest.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertFalse(any(r['source_id'] == 'session_A' and r['image'] == '000002.jpg' for r in rows))
        self.assertEqual(before, {p: p.read_bytes() for source in self.paths for p in source.rglob('*') if p.is_file()})

    def test_cross_session_duplicates_are_removed_before_splitting(self):
        self.approve()
        first = Path(self.sources[0]['path']) / 'images/000002.jpg'
        second = Path(self.sources[1]['path']) / 'images/000002.jpg'
        second.write_bytes(first.read_bytes())
        Path(self.sources[1]['review']).unlink()
        store = ReviewStore(self.sources[1]['path'], self.sources[1]['review'])
        store.update({'revision': 0, 'indices': [1, 2, 3], 'changes': {'decision': 'keep'}})
        out = self.root / 'out'
        report = prepare.build_dataset(self.paths, out)
        self.assertEqual(report['audit']['duplicate_copies_removed'], 1)
        with (out / 'manifest.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len({r['sha256'] for r in rows}))
        self.assertTrue(any('session_A/' in r['duplicate_sources'] and 'session_B/' in r['duplicate_sources'] for r in rows))

    def test_label_override_and_original_are_both_in_manifest(self):
        self.approve()
        item = self.sources[0]
        store = ReviewStore(item['path'], item['review'])
        store.update({'revision': 0, 'indices': [3], 'changes': {'decision': 'keep', 'label_override': 'RIGHT'}})
        out = self.root / 'out'
        prepare.build_dataset(self.paths, out)
        with (out / 'manifest.csv').open() as f:
            rows = list(csv.DictReader(f))
        changed = next(r for r in rows if r['action'] == 'LEFT' and r['source_id'] == 'session_A')
        self.assertEqual(changed['label'], 'RIGHT')

    def test_same_session_duplicates_are_deduplicated(self):
        self.approve()
        item = self.sources[0]
        source = Path(item['path'])
        (source / 'images/000004.jpg').write_bytes((source / 'images/000002.jpg').read_bytes())
        Path(item['review']).unlink()
        store = ReviewStore(item['path'], item['review'])
        store.update({'revision': 0, 'indices': [1, 2, 3],
                      'changes': {'decision': 'keep'}})
        store.update({'revision': 1, 'indices': [3], 'changes': {'label_override': 'RIGHT'}})
        out = self.root / 'out'
        stats = prepare.build_dataset(self.paths, out)
        self.assertEqual(stats['audit']['duplicate_copies_removed'], 1)
        self.assertEqual(len(list(out.rglob('*.jpg'))), 8)

    def test_conflicting_labels_for_identical_content_block_build(self):
        self.approve()
        item = self.sources[0]
        source = Path(item['path'])
        (source / 'images/000003.jpg').write_bytes((source / 'images/000002.jpg').read_bytes())
        Path(item['review']).unlink()
        store = ReviewStore(item['path'], item['review'])
        store.update({'revision': 0, 'indices': [1, 2, 3], 'changes': {'decision': 'keep'}})
        with self.assertRaisesRegex(ValueError, 'conflicting labels'):
            prepare.build_dataset(self.paths, self.root / 'out')

    def test_missing_class_is_reported_for_training_time_selection(self):
        self.approve()
        item = self.sources[2]
        store = ReviewStore(item['path'], item['review'])
        store.update({'revision': 0, 'indices': [2], 'changes': {'decision': 'exclude'}})
        stats = prepare.build_dataset(self.paths, self.root / 'out')
        self.assertIn('test/STRAIGHT', stats['empty_classes'])
        with self.assertRaisesRegex(ValueError, 'test/STRAIGHT'):
            prepare.validate_split_classes(self.root / 'out', 'test', ['RIGHT', 'STRAIGHT'])

    def test_default_build_preserves_three_classes_for_later_training_selection(self):
        self.approve()
        for item in self.sources:
            store = ReviewStore(item['path'], item['review'])
            store.update({'revision': 0, 'indices': [3], 'changes': {'decision': 'keep'}})
        out = self.root / 'all_classes'
        prepare.build_dataset(self.paths, out)
        self.assertEqual(prepare.read_classes(out), ['LEFT', 'RIGHT', 'STRAIGHT'])
        self.assertEqual(prepare.read_classes(out, ['RIGHT', 'STRAIGHT']), ['RIGHT', 'STRAIGHT'])
        self.assertTrue((out / 'train/LEFT').is_dir())
        self.assertEqual(len(list(out.rglob('*.jpg'))), 9)
        for invalid in (['RIGHT'], ['RIGHT', 'RIGHT'], ['STOP', 'RIGHT']):
            with self.assertRaises(ValueError):
                prepare.read_classes(out, invalid)

    def test_single_session_can_be_pooled_and_split_without_config_json(self):
        source = self.root / 'single_session'
        (source / 'images').mkdir(parents=True)
        rows = []
        for index, original in enumerate(self.paths):
            with (original / 'labels.csv').open() as handle:
                for row in csv.DictReader(handle):
                    old_name = row['image']
                    row['image'] = f'{index}_{old_name}'
                    (source / 'images' / row['image']).write_bytes((original / 'images' / old_name).read_bytes())
                    rows.append(row)
        with (source / 'labels.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=prepare.SOURCE_FIELDS[:4])
            writer.writeheader()
            writer.writerows(rows)
        store = ReviewStore(source)
        store.update({'revision': 0, 'indices': [i for i, r in enumerate(store.rows) if r['action'] != 'STOP'],
                      'changes': {'decision': 'keep'}})
        before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
        output = self.root / 'single_output'
        report = prepare.build_dataset([source], output)
        self.assertTrue(all(report['counts'][split][label] == 1 for split in prepare.SPLITS for label in prepare.CLASSES))
        self.assertEqual(sorted(p.name for p in output.glob('*.json')), ['dataset.json'])
        self.assertEqual(before, {p: p.read_bytes() for p in source.rglob('*') if p.is_file()})

    def test_model_class_order_and_legacy_fallback(self):
        self.assertEqual(prepare.read_classes(self.root), list(prepare.CLASSES))
        prepare.write_json(self.root / 'labels.json', ['RIGHT', 'STRAIGHT'])
        prepare.check_model_classes(self.root, ['RIGHT', 'STRAIGHT'])
        with self.assertRaisesRegex(ValueError, 'differ'):
            prepare.check_model_classes(self.root, ['STRAIGHT', 'RIGHT'])

    def test_external_review_edits_are_not_overwritten(self):
        item = self.sources[0]
        store = ReviewStore(item['path'], item['review'])
        rows = [dict(r) for r in store.rows]
        rows[0]['decision'] = 'exclude'
        prepare.save_review(item['review'], rows)
        with self.assertRaisesRegex(ValueError, 'externally'):
            store.update({'revision': 0, 'indices': [1], 'changes': {'decision': 'keep'}})


if __name__ == '__main__':
    unittest.main()
