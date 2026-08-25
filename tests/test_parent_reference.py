import arterial_spectral_cascade.parent as parent


def test_legacy_topology_is_diagnostic_not_acceptance(monkeypatch):
    response = {2.0:0.1,5.0:0.3,10.0:1.0,15.0:2.0,20.0:3.0}
    def fake_run_case(prep, paths=None, resume=True, progress=True):
        wo=float(prep.spec.Wo0)
        return {"summary":{"R_max":response[wo],"s_R_max":1.0,"I2_final":1.0,"G_bal_max":-1e-3,
                           "eta_tail_max":1e-12,"max_abs_RI1_inst":1e-12,"max_abs_RE_inst":1e-12,
                           "completed":True,"runtime_valid":True}}
    monkeypatch.setattr(parent,"run_case",fake_run_case)
    result=parent.run_parent_reference_audit(paths=None,N=32,dt=0.01,T_final=0.1,progress=False)
    assert result["solver_design_parent_baseline"]["pass_numerical"]
    assert result["legacy_reference_audit"]["topology_match"] is False
    assert result["legacy_reference_audit"]["acceptance_controlling"] is False
    assert result["pass"] is True
