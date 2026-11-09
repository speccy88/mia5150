"""Package completed evidence and replay it from the deliverable files."""
from common import STUDY, read, save, sha
import struct
import subprocess
import sys
from xml.etree import ElementTree


def cpu(script, *args):
    subprocess.run([sys.executable, str(STUDY / 'src' / script), *args], check=True)


def main():
    report = STUDY / 'REPORT.md'
    lock = STUDY / 'FINAL_EVAL_LOCK.json'
    assert report.exists() and read(lock)['locked']
    completed = read(STUDY / 'results/FINAL_EXECUTION_COMPLETE.json')
    assert completed['status'] == 'complete' and completed['evaluation_lock_sha256'] == sha(lock)
    assert completed['report_sha256'] == sha(report)
    assert completed['vector_archive_manifest_sha256'] == sha(STUDY / 'vectors/manifest.json')
    archive = STUDY / 'vectors/dense-vectors-v1.npz'
    assert sha(archive) == read(STUDY / 'vectors/manifest.json')['archive_sha256']
    assert not (STUDY / 'FILE_MANIFEST.json').exists(), 'Publication is already frozen'
    package_manifest = STUDY / 'manifests/record_archive_manifest.json'
    if not package_manifest.exists():
        cpu('copy_run_receipts.py')
        cpu('package_records.py')
    for name, info in read(package_manifest)['files'].items():
        path = STUDY / name
        assert path.stat().st_size == info['bytes'] and sha(path) == info['sha256'], name
    cpu('verify.py')
    cpu('snapshot.py', 'replay')
    replay = STUDY / 'results/vector_replay.json'
    verify = STUDY / 'results/independent_verification.json'
    assert read(replay)['status'] == read(verify)['status'] == 'passed'
    figures = {}
    for png in sorted((STUDY / 'figures').glob('*.png')):
        with png.open('rb') as f:
            header = f.read(24)
        assert header.startswith(b'\x89PNG\r\n\x1a\n') and header[12:16] == b'IHDR'
        width, height = struct.unpack('>II', header[16:24])
        svg = png.with_suffix('.svg')
        assert width > 100 and height > 100 and ElementTree.parse(svg).getroot().tag.endswith('}svg')
        figures[png.stem] = {'png_sha256': sha(png), 'svg_sha256': sha(svg), 'pixel_size': [width, height]}
    assert len(figures) >= 7
    save(STUDY / 'results/publication_readiness.json', {
        'status': 'passed', 'lock_sha256': sha(lock), 'report_sha256': sha(report),
        'vector_replay_sha256': sha(replay), 'independent_verification_sha256': sha(verify),
        'record_archive_manifest_sha256': sha(package_manifest),
        'figure_file_checks': figures,
    })
    print('Packaged and independently replayed the complete local evidence; ready for publication review.', flush=True)


if __name__ == '__main__':
    main()
