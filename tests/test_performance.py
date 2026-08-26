import zipfile

import numpy as np

import arterial_spectral_cascade as asc
import arterial_spectral_cascade.performance as perf
import arterial_spectral_cascade.study as study
from arterial_spectral_cascade.storage import save_checkpoint, _record, _history_append
from arterial_spectral_cascade.core import project_hat, initial_condition, prepare_case, CaseSpec


def test_optimized_backend_is_reference_verified():
    status=asc.PERFORMANCE_BACKEND_STATUS
    assert status.verified
    assert status.active in {"optimized","reference"}
    check=perf.verify_optimized_backend_equivalence()
    assert check["pass"]
    assert check["trajectory_error"] < 5e-12


def test_class_specific_converged_settings_are_used_when_unspecified():
    cfg=dict(study.STUDY_CONFIG)
    cfg["_CLASS_NUMERICAL_SETTINGS"]={"DL":{"N":128,"dt":4e-4,"source_case_id":"dl-hard"},
                                      "DM":{"N":256,"dt":2e-4,"source_case_id":"dm-hard"}}
    record={"case_id":"dl-test","case_class":"DL","chi_b":0.01,"chi_g":0.01,"w":3.0,"p":1}
    spec=study.case_record_to_spec(record,10.0,cfg=cfg)
    assert spec.N==128
    assert spec.dt==4e-4
    explicit=study.case_record_to_spec(record,10.0,N=64,dt=1e-3,cfg=cfg)
    assert explicit.N==64
    assert explicit.dt==1e-3


def test_checkpoint_format_matches_active_backend(tmp_path):
    spec=CaseSpec("P1",Wo0=5,N=32,dt=.002,T_final=.02,k0=.5,eps_b=.05,eps_g=.04,q=1.0,
                  output_every_steps=2,checkpoint_every_steps=5)
    prep=prepare_case(spec)
    paths=asc.init_project_paths(tmp_path)
    ah=project_hat(np.fft.fft(initial_condition(spec,prep.grid)),prep.grid)
    history={}; rec=_record(prep,ah,0); _history_append(history,rec)
    peak={"R":rec["R"],"step":0,"ahat":ah.copy()}
    path=save_checkpoint(prep,paths,0,ah,history,peak)
    expected=(zipfile.ZIP_STORED if asc.PERFORMANCE_BACKEND_STATUS.active=="optimized"
              else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(path,"r") as z:
        assert z.infolist()
        assert all(info.compress_type==expected for info in z.infolist())
