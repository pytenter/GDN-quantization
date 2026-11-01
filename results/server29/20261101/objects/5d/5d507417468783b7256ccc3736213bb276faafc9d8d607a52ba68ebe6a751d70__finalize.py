from pathlib import Path

from common import ARMS, CONDITIONS, EXP, read_json, sha256, write_json, write_text


def main():
    result=read_json(EXP/'FINAL_STATUS.json')
    if result['DISCOVERY_COMPLETE']!='32/32' or result['FINAL_ACTION']!='STOP':
        raise RuntimeError('final scientific status incomplete')
    for doc in range(1,5):
        release=read_json(EXP/f'discovery/doc_{doc:02d}/drive/drive_release_record.json')
        if not release['release_complete']:
            raise RuntimeError('drive release incomplete')
        for condition in CONDITIONS:
            for arm in ARMS:
                path=EXP/f'discovery/unit_results/doc_{doc:02d}_{condition}_{arm}.json'
                if read_json(path)['status']!='PASS':
                    raise RuntimeError('unit result incomplete')
    excluded={'SHA256SUMS','FINAL_SHA256SUMS','FINAL_ARTIFACT_INTEGRITY.json'}
    rows=[]
    for path in sorted(p for p in EXP.rglob('*') if p.is_file() and p.name not in excluded and not p.name.endswith('.tmp')):
        rows.append((sha256(path),str(path.relative_to(EXP))))
    write_text(EXP/'SHA256SUMS',''.join(f'{h}  {p}\n' for h,p in rows))
    write_json(EXP/'FINAL_ARTIFACT_INTEGRITY.json',{
        'FINAL_ARTIFACT_INTEGRITY_GATE':'PASS','artifact_count':len(rows),
        'sha256_manifest_sha256':sha256(EXP/'SHA256SUMS'),
        'all_32_units_present':True,'all_drives_released':True})
    rows2=[]
    for path in sorted(p for p in EXP.rglob('*') if p.is_file() and p.name!='FINAL_SHA256SUMS' and not p.name.endswith('.tmp')):
        rows2.append((sha256(path),str(path.relative_to(EXP))))
    write_text(EXP/'FINAL_SHA256SUMS',''.join(f'{h}  {p}\n' for h,p in rows2))
    print('FINAL_ARTIFACT_INTEGRITY_PASS_STOP',flush=True)


if __name__=='__main__':
    main()
