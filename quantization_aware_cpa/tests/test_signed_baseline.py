"""Small CPU gates for wire format, protocol isolation, and unchanged evaluation."""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest
import torch

from qacpa import signed_baseline as runner
from qacpa import signed_baseline_observation as wire
from qacpa.full_update_evaluation import evaluate_candidate


@pytest.fixture(autouse=True)
def small_cpu_pool():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("G", [torch.arange(-7.,8.), torch.tensor([0.]), torch.zeros(2,3),
                               torch.tensor([-1e-30,0.,1e-30]), torch.tensor([1.7500007152557373])])
def test_wire_reference_exact_roundtrip(G):
    q,dq,s,_ = runner.quantize_tensor_symmetric(G,bits=4,rounding="nearest")
    packet = wire.pack(q,s)
    aq,ass = wire.unpack(packet,tuple(G.shape))
    assert torch.equal(q,aq) and torch.equal(s,ass)
    assert torch.equal(dq,runner.cell_utils.dequantize(aq,ass))
    assert len(packet)==20+(G.numel()+1)//2


@pytest.mark.parametrize("damage", ["reserved", "padding", "affine", "length", "scale"])
def test_invalid_packet_rejected(damage):
    packet = bytearray(wire.pack(torch.tensor([1],dtype=torch.int8),torch.tensor(1.)))
    if damage=="reserved": packet[-1]=15
    if damage=="padding": packet[-1]=0x18
    if damage=="affine": packet[:8]=b"AFFINE00"
    if damage=="length": packet.append(0)
    if damage=="scale": packet[16:20]=b"\0\0\0\0"
    with pytest.raises(ValueError): wire.unpack(bytes(packet),(1,))


@pytest.fixture
def tiny_bundle(tmp_path,monkeypatch):
    shapes = {"classifier.0.weight": (3,5), "classifier.0.bias": (3,)}
    monkeypatch.setattr(wire,"FC_SHAPES",shapes)
    updates = {name:torch.linspace(-1,1,steps=int(torch.tensor(shape).prod())).reshape(shape) for name,shape in shapes.items()}
    output = tmp_path/"public"
    manifest = wire.write_bundle(output,updates)
    return output,manifest,updates


def test_public_bundle_only_q_scale_context(tiny_bundle):
    folder,manifest,updates = tiny_bundle
    actual,entries = wire.read_bundle(folder)
    assert actual==manifest
    for name,G in updates.items():
        q,s = runner.cell_utils.encode(G)
        assert torch.equal(entries[name][0],q) and torch.equal(entries[name][1],s)
    text = json.dumps(manifest)
    for field in ('"max_abs"','"x"','"z"','"updates"','"private_error"'):
        assert field not in text
    assert manifest["communication"]["total_file_bytes"]==sum(p.stat().st_size for p in folder.iterdir())


@pytest.mark.parametrize("damage", ["truth_field","codec","scope","path","shape","hash"])
def test_manifest_rejects_private_or_incompatible_fields(tiny_bundle,damage):
    folder,manifest,_ = tiny_bundle
    manifest = deepcopy(manifest)
    layer = manifest["layers"]["classifier.0.weight"]
    if damage=="truth_field": manifest["x"]="private.pt"
    if damage=="codec": manifest["format"]="uniform_affine_uint4_nearest_even_v1"
    if damage=="scope": manifest["codec_metadata"]["scale_scope"]="per_row"
    if damage=="path": layer["file"]="../private.pt"
    if damage=="shape": layer["shape"]=[15]
    if damage=="hash": layer["sha256"]="0"*64
    (folder/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    with pytest.raises(ValueError): wire.read_bundle(folder)


def test_naive_has_no_private_api_and_uses_raw_sq(monkeypatch,tmp_path):
    assert set(inspect.signature(runner.run_naive).parameters)=={"public_observation","output","checkpoint"}
    q=torch.tensor([[-7,0,7],[1,2,3]],dtype=torch.int8)
    s=torch.tensor(.002)
    monkeypatch.setattr(runner,"read_bundle",lambda path: ({"checkpoint_sha256":wire.CHECKPOINT_SHA},{runner.ATTACK_LAYER:(q,s)}))
    monkeypatch.setattr(runner,"digest",lambda path:"public_digest")
    captured={}
    def run(**kwargs): captured.update(kwargs); return {"status":"COMPLETED"}
    monkeypatch.setattr(runner,"_run_arm",run)
    monkeypatch.setattr(torch,"load",lambda *a,**kw:pytest.fail("naive tried to load private tensor file"))
    runner.run_naive(public_observation=tmp_path,output=tmp_path/"out",checkpoint=tmp_path/"ckpt")
    assert torch.equal(captured["Y"],q.float()*s)
    assert captured["input_info"]["private_inputs_used"] is False


def test_reference_uses_raw_delta_without_sign_or_lr_scaling(monkeypatch,tmp_path):
    G=torch.tensor([[-.1,0.,.02]])
    path=tmp_path/"raw.pt"
    torch.save({"updates":{runner.ATTACK_LAYER:G},"checkpoint_sha256":wire.CHECKPOINT_SHA,
                "delta_convention":"theta_after10_minus_theta_start"},path)
    captured={}
    monkeypatch.setattr(runner,"_run_arm",lambda **kw:captured.update(kw))
    runner.run_reference(raw_updates=path,output=tmp_path/"ref",checkpoint=tmp_path/"ckpt")
    assert torch.equal(captured["Y"],G)


def test_fixed_two_arm_settings_and_no_private_fia_api():
    assert runner.CPA_CONFIG["cpa_ica_max_iter"]==runner.FIA_CONFIG["n_iter"]==25000
    assert runner.CPA_CONFIG["cpa_ica_scheduler"]=="none"
    assert runner.FIA_CONFIG["scheduler"]=="cosine"
    assert runner.SEED==42 and runner.CPA_CONFIG["cpa_ica_ne"]==1.
    assert set(inspect.signature(runner.upstream_naive.run_upstream_fia).parameters)=={
        "target_z","device","config","output_dir","seed"}
    assert not {"images","labels","truth","raw_updates"} & set(inspect.signature(runner._run_arm).parameters)
    with pytest.raises(SystemExit):
        runner.parser().parse_args(["naive","--public-observation","a","--output","b","--checkpoint","c","--truth-source","secret"])


def test_label_free_image_evaluation_preserves_existing_matching(tmp_path):
    pixels = torch.stack((torch.full((3,224,224),.2),torch.full((3,224,224),.8)))
    gt_pixels = pixels.flip(0)*.9
    mean = pixels.new_tensor((.485,.456,.406)).reshape(1,3,1,1)
    std = pixels.new_tensor((.229,.224,.225)).reshape(1,3,1,1)
    p,t=tmp_path/"candidate.pt",tmp_path/"truth.pt"
    torch.save({"pixels":pixels,"labels":torch.tensor([1,2])},p)
    torch.save({"x":(gt_pixels-mean)/std,"labels":torch.tensor([2,1])},t)
    default=evaluate_candidate(p,t)
    torch.save({"pixels":pixels},p)
    torch.save({"x":(gt_pixels-mean)/std},t)
    image_only=evaluate_candidate(p,t,include_labels=False)
    for key in ("mean_mse","mean_psnr_db","mean_ssim","matching"):
        assert image_only[key]==default[key]
    for p1,p2 in zip(image_only["per_image"],default["per_image"]):
        assert p1=={k:v for k,v in p2.items() if k!="label_correct"}
    assert "label_accuracy" not in image_only
    with pytest.raises(ValueError): evaluate_candidate(p,t)


def test_offline_requires_both_final_hashes_before_truth(tmp_path):
    from qacpa.signed_baseline_evaluation import _completed
    (tmp_path/"result.json").write_text(json.dumps({"status":"RUNNING","cpa_iterations":25000,"fia_iterations":1}),encoding="utf-8")
    with pytest.raises(ValueError,match="complete"):_completed(tmp_path)


def test_embedding_matching_offline_distinct_from_image_matching():
    from qacpa.signed_baseline_evaluation import embedding_evaluation
    z=torch.eye(3)
    result=embedding_evaluation(z[[2,0,1]],z)
    assert [r["truth_index"] for r in result["per_source"]]==[2,0,1]
    assert result["mean_cosine"]==1.


def test_source_metric_does_not_claim_actual_ne_or_mi():
    S=torch.tensor([[1.,-2.,0.],[0.,1.,-1.]])
    record=runner.source_metrics(S,[{"loss":4.,"loss_ne":0.,"loss_l1":1.,"loss_nv":.5}])
    for key in ("negentropy","mi"):
        assert record["raw_metrics"][key]["value"] is None
    assert record["evaluation_metrics"]["image_oracle"]["lpips"]["value"] is None
