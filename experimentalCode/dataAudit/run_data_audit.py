"""Run the complete, read-only data audit from any working directory.

Usage: python experimentalCode/dataAudit/run_data_audit.py
Required: numpy, pandas, openpyxl. No network, model fitting or raw data writes.
"""
from pathlib import Path
import contextlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataAudit.audit_data as audit_data
import dataAudit.audit_deep as audit_deep
import dataAudit.audit_supplement as audit_supplement
import dataAudit.build_audit_report as build_audit_report


def main():
    root=Path(__file__).resolve().parents[2]
    out=root/'experimentalCode'/'results'/'data_audit';out.mkdir(parents=True,exist_ok=True)
    # audit_data accepts its own CLI; ignore wrapper arguments consistently.
    sys.argv=[str(Path(audit_data.__file__).resolve()),'--root',str(root)]
    with (out/'audit_run_log.txt').open('w',encoding='utf-8') as log:
        with contextlib.redirect_stdout(log):
            audit_data.main()
            audit_deep.main()
            audit_supplement.main()
            build_audit_report.main()
    print('Audit completed. Raw inputs were not modified.')
    print(out/'啤酒风味数据审计与附件说明.md')


if __name__=='__main__':main()
