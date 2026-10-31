"""Read-only source checks and document previews for the selected CIFAR figure."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_assets"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    manifest = json.loads((OUT / "cifar100_selection_manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection"]["display_ordinals_one_based"] == [11, 24]
    for group in ("sources_sha256", "outputs_sha256"):
        for relative, digest in manifest[group].items():
            assert sha(ROOT / relative) == digest, relative
    for record in manifest["cifar_tiles"]:
        method = record["method"].replace("_R8", "")
        prefix = f"example{record['display_ordinal_one_based']:02d}"
        filename = f"{prefix}_original.png" if method == "original" else f"{prefix}_bits{record['bits']}_{method}.png"
        with Image.open(ROOT / record["source"]) as image:
            expected = np.asarray(image.convert("RGB").crop(record["crop_xyxy"]))
        with Image.open(OUT / "cifar100_selected_examples11_24" / filename) as image:
            assert np.array_equal(np.asarray(image), expected)
        assert hashlib.sha256(expected.tobytes()).hexdigest() == record["rgb_sha256"]
    protected = {
        "paper_assets/combined_results.tex": "d4851bf38cb1d8bbbbf7b14c4f8e33c9f8efc0b1959d2c6a653cf49803386fa5",
        "test.tex": "e43273f70736aab00769cf66b854880212dc0305933b8b8038207d1dd105928d",
        "paper_assets/sparsification_naive_random.pdf": "21c34fad391b844f82185f3deae338a7b7cfc7eeaee543c7ba5fb6b507d555fa",
    }
    for relative, digest in protected.items():
        assert sha(ROOT / relative) == digest, f"Unrelated file changed: {relative}"
    tex = (ROOT / "test_cora.tex").read_text(encoding="utf-8")
    assert "paper_assets/quantization_cifar100_imagenet.pdf" in tex
    assert "CIFAR-100 image panels are not shown." not in tex
    assert "selected post hoc" in tex
    for relative in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        assert (ROOT / relative).is_file(), relative

    from pypdf import PdfReader
    pdf = PdfReader(ROOT / "test_cora.pdf")
    renderer = shutil.which("pdftoppm")
    assert renderer, "A PDF preview renderer is required"
    preview = ROOT / "paper_preview/cifar100_examples11_24"
    preview.mkdir(parents=True, exist_ok=True)
    def render(index, target):
        subprocess.run([renderer, "-f", str(index+1), "-l", str(index+1),
                        "-singlefile", "-r", "108", "-png",
                        str(ROOT / "test_cora.pdf"), str(target.with_suffix(""))],
                       check=True, timeout=60,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    figure_pages = []
    sparse_pages = []
    for index, page in enumerate(pdf.pages):
        text = page.extract_text()
        if "Fig.1.Pairedreconstructions" in "".join(text.split()):
            target = preview / f"figure_page_{index+1}.png"
            render(index, target)
            figure_pages.append({"page": index+1, "preview": target.relative_to(ROOT).as_posix()})
        if "Fig.2.Pairedsparsi" in "".join(text.split()):
            target = preview / f"sparsification_page_{index+1}.png"
            render(index, target)
            sparse_pages.append({"page": index+1, "preview": target.relative_to(ROOT).as_posix()})
    count = len(pdf.pages)
    assert len(figure_pages) == 1, "Combined figure must occur exactly once"
    if "paper_assets/sparsification_cifar100_imagenet.pdf" in tex:
        assert len(sparse_pages) == 1
        sparse_manifest = json.loads((OUT / "cifar100_sparse_manifest.json").read_text(encoding="utf-8"))
        for group in ("sources_sha256", "outputs_sha256"):
            for relative, digest in sparse_manifest[group].items():
                assert sha(ROOT / relative) == digest, relative
    log = (ROOT / "test_cora.log").read_text(encoding="utf-8", errors="replace")
    assert "undefined references" not in log.lower()
    result = {
        "status": "PASS", "selected_examples": [11, 24],
        "source_hashes_checked": len(manifest["sources_sha256"]),
        "all_exported_tiles_pixel_equal": True,
        "protected_files_unchanged": list(protected),
        "paper_pages": count, "figure_pages": figure_pages,
        "sparsification_figure_pages": sparse_pages,
        "pdf_sha256": sha(ROOT / "test_cora.pdf"),
        "preexisting_table_warnings": [
            "combined_results.tex has two overfull alignments (~18.32pt)",
            "combined_results.tex contains 0ㅇㅇ323 in ImageNet sparse LPIPS; left unchanged"
        ],
    }
    (OUT / "cifar100_document_validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
