# -*- coding: utf-8 -*-
# Generate diagnostics/ocr_benchmark/transcribe_data.js from the GT + answers JSONs.
# Run: python diagnostics/ocr_benchmark/gen_data_js.py
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
GT_FILE = HERE / 'ground_truth_readable.json'
ANS_FILE = HERE / 'review_sheets' / 'transcript_answers.json'
OUT_FILE = HERE / 'transcribe_data.js'


def main() -> None:
    gt = json.loads(GT_FILE.read_text(encoding='utf-8'))
    plates = gt['plates']
    ans = json.loads(ANS_FILE.read_text(encoding='utf-8')).get('answers', {})

    ids = []
    entries = []
    for pid, p in plates.items():
        idx = len(ids)
        ids.append(pid)
        d = ans.get(str(idx), {}) or {}
        crop_path = p.get('crop_path') or ''
        entries.append({
            'id': pid,
            'crop_path': crop_path,
            'crop_file': Path(crop_path).name,
            'source_image': p.get('source_image') or '',
            'layout': p.get('layout') or '',
            'crop_w': int(p.get('crop_width') or 0),
            'crop_h': int(p.get('crop_height') or 0),
            'crop_ar': float(p.get('crop_aspect_ratio') or 0.0),
            'yolo_conf': float(p.get('yolo_confidence') or 0.0),
            'ocr_text': p.get('system_ocr_text') or '',
            'gt_text': p.get('ground_truth_text') or '',
            'status': p.get('status') or '',
            'verified': bool(p.get('verified')),
            'notes': p.get('notes') or '',
            'hint': d.get('hint') or p.get('system_ocr_text') or '',
        })

    draft = {}
    for i, pid in enumerate(ids):
        d = ans.get(str(i), {}) or {}
        draft[str(i)] = {
            'text': d.get('text') or '',
            'unreadable': bool(d.get('unreadable')),
            'verified': bool(entries[i]['verified']),
            'review': False,
        }

    data = {
        'generated': '2026-10-10T00:00:00Z',
        'total': len(ids),
        'ids': ids,
        'entries': entries,
        'draft': draft,
    }

    body = json.dumps(data, indent=2, ensure_ascii=False)
    OUT_FILE.write_text('window.PLATE_DATA = ' + body + ';\n', encoding='utf-8')
    print('wrote', OUT_FILE, 'total entries:', len(ids))


if __name__ == '__main__':
    main()
