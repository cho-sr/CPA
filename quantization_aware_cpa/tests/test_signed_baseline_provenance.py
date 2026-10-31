"""Planner review 01: offline provenance/output rejection, synthetic files only."""
from copy import deepcopy
import json

import pytest
import torch

from qacpa import signed_baseline_evaluation as ev


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf-8")


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    public,private = tmp_path/"public",tmp_path/"private"
    monkeypatch.setattr(ev,"PUBLIC_ROOT",public)
    monkeypatch.setattr(ev,"PRIVATE_ROOT",private)
    paths={"naive":public/"naive","reference":private/"reference",
           "public_observation":public/"observation","private_observation":private/"observation",
           "output":private/"evaluation"}
    for k,p in paths.items():
        if k!="output": p.mkdir(parents=True)
    manifest=paths["public_observation"]/"manifest.json"
    write(manifest,{"checkpoint_sha256":ev.CHECKPOINT_SHA})
    raw=paths["private_observation"]/"raw_updates.pt"
    truth=paths["private_observation"]/"evaluation_truth.pt"
    raw.write_bytes(b"synthetic raw bytes")
    truth.write_bytes(b"synthetic truth bytes; must not be torch.load'ed by validation")
    write(paths["private_observation"]/"sender_validation.json",{
        "status":"COMPLETED","checkpoint_sha256":ev.CHECKPOINT_SHA,"source_truth_sha256":ev.TRUTH_SHA,
        "raw_updates_sha256":ev.digest(raw),"public_manifest_sha256":ev.digest(manifest),
        "evaluation_truth_sha256":ev.digest(truth)})
    source_hashes=ev.upstream_naive.source_hashes()
    for arm in ("naive","reference"):
        folder=paths[arm]
        (folder/"candidate.pt").write_bytes(b"synthetic final image bytes")
        (folder/"sources.pt").write_bytes(b"synthetic final source bytes")
        inp=({"arm":"INT4_naive","public_manifest_sha256":ev.digest(manifest),
              "checkpoint_sha256":ev.CHECKPOINT_SHA,"private_inputs_used":False} if arm=="naive" else
             {"arm":"FP32 Reference","raw_file_sha256":ev.digest(raw),"private_raw_update_used":True,"oracle_truth_used":False})
        write(folder/"result.json",{"status":"COMPLETED","cpa_iterations":25000,"fia_iterations":25000,
              "input":inp,"seed":42,"image_count":16,"labels_used":False,
              "selection":"final_iteration_only_no_oracle_selection",
              "candidate_sha256":ev.digest(folder/"candidate.pt"),"sources_sha256":ev.digest(folder/"sources.pt")})
        write(folder/"registered_arm.json",{"input":inp,"seed":42,"checkpoint_sha256":ev.CHECKPOINT_SHA,
              "cpa":deepcopy(ev.CPA_CONFIG),"fia":deepcopy(ev.FIA_CONFIG),
              "input_conversion":"identity_raw_Delta_or_raw_s_q; no sign/lr/T/B transformation",
              "source_adapter":"abs(S_hat); estimated ordering; no oracle correction",
              "source_hashes":source_hashes,"input_tensor_sha256":"fake_tensor_hash"})
        for kind in ("cpa","fia"):
            write(folder/f"{kind}_result.json",{"source_hashes":source_hashes,"settings":{"seed":42}})
        write(folder/"cpa/cpa_manifest.json",{"fingerprint":{"input_sha256":"fake_tensor_hash"},
                                               "cache_root":str(folder/"cpa/cache")})
    monkeypatch.setattr(torch,"load",lambda *a,**kw:pytest.fail("guard tried to load private tensors"))
    return paths


def test_review_accepts_correct_distinct_bound_inputs_without_truth_load(bundle):
    results,sender,audit=ev.validate_provenance(**bundle)
    assert audit["status"]=="PASS" and audit["private_output_isolated"]
    assert results["reference"]["input"]["raw_file_sha256"]==sender["raw_updates_sha256"]
    assert not bundle["output"].exists()


@pytest.mark.parametrize("mode",["duplicated","swapped","public_observation_output","naive_output",
                                 "private_root_output","overlapping_reference_output","outside_private"])
def test_review_rejects_wrong_paths_before_any_truth_read_or_write(bundle,mode):
    args=dict(bundle)
    if mode=="duplicated": args["reference"]=args["naive"]
    if mode=="swapped": args["naive"],args["reference"]=args["reference"],args["naive"]
    if mode=="public_observation_output": args["output"]=args["public_observation"]/"evaluation"
    if mode=="naive_output": args["output"]=args["naive"]/"evaluation"
    if mode=="private_root_output": args["output"]=args["reference"].parent
    if mode=="overlapping_reference_output": args["output"]=args["reference"]/"evaluation"
    if mode=="outside_private": args["output"]=args["reference"].parent.parent/"elsewhere"
    with pytest.raises(ValueError): ev.validate_provenance(**args)
    assert not bundle["output"].exists()


@pytest.mark.parametrize("damage",["same_identity","raw_input","public_input","sender_raw","sender_public",
    "checkpoint","seed","cpa","fia","conversion","adapter","source_version","executed_source","cache","fingerprint"])
def test_review_rejects_provenance_mismatch(bundle,damage):
    folder=bundle["reference"]
    filename="registered_arm.json"
    if damage in ("same_identity","raw_input"): filename="result.json"
    if damage=="public_input": folder=bundle["naive"]; filename="result.json"
    if damage in ("sender_raw","sender_public"): folder=bundle["private_observation"]; filename="sender_validation.json"
    if damage=="executed_source": filename="cpa_result.json"
    if damage in ("cache","fingerprint"): filename="cpa/cpa_manifest.json"
    path=folder/filename
    data=json.loads(path.read_text())
    if damage=="same_identity": data["input"]["arm"]="INT4_naive"
    elif damage=="raw_input": data["input"]["raw_file_sha256"]="0"*64
    elif damage=="public_input": data["input"]["public_manifest_sha256"]="0"*64
    elif damage=="sender_raw": data["raw_updates_sha256"]="0"*64
    elif damage=="sender_public": data["public_manifest_sha256"]="0"*64
    elif damage=="checkpoint": data["checkpoint_sha256"]="0"*64
    elif damage=="seed": data["seed"]=43
    elif damage=="cpa": data["cpa"]["cpa_ica_max_iter"]=1
    elif damage=="fia": data["fia"]["n_iter"]=1
    elif damage=="conversion": data["input_conversion"]="-Delta/lr"
    elif damage=="adapter": data["source_adapter"]="oracle_permutation"
    elif damage in ("source_version","executed_source"): data["source_hashes"]={}
    elif damage=="cache": data["cache_root"]=str(bundle["naive"])
    elif damage=="fingerprint": data["fingerprint"]["input_sha256"]="other_input"
    write(path,data)
    with pytest.raises(ValueError): ev.validate_provenance(**bundle)
    assert not bundle["output"].exists()


@pytest.mark.parametrize("filename",["raw_updates.pt","evaluation_truth.pt"])
def test_review_rejects_changed_actual_private_bytes(bundle,filename):
    (bundle["private_observation"]/filename).write_bytes(b"altered bytes")
    with pytest.raises(ValueError): ev.validate_provenance(**bundle)
    assert not bundle["output"].exists()
