# Third-party notices and acquisition boundaries

**MIT LICENSE DOES NOT APPLY TO THIRD-PARTY MATERIALS.** The root MIT grant
covers only author-owned RAIL-3 original code and its associated code-use
materials as described in [LICENSE_STATUS.md](LICENSE_STATUS.md).

No upstream software, wheels, checkpoints, raw benchmarks, photographs, full
manuscript, reference PDFs or standalone font files are redistributed here.
The table retains previously verified version-bound source notices. Dependencies
are third-party software, not RAIL-3 original code. Upstream terms that have not
been reliably verified are marked **USER_MUST_REVIEW_UPSTREAM_TERMS**; a source
notice is not certification of every installed wheel or transitive component.

| Component / recorded version | Upstream terms / source | Distribution boundary |
| --- | --- | --- |
| Python (release requires >=3.12) | [Official Python licence and history](https://docs.python.org/3/license.html); exact runtime bundle: USER_MUST_REVIEW_UPSTREAM_TERMS | Interpreter dependency; not vendored or covered by the RAIL-3 MIT grant. |
| SAM 3.1 backend: sam3.1-object-multiplex; source commit 96914d2425f90a64f45ca977c2b5165418099543 | SAM License; custom agreement, last updated November 19, 2025; [official licence](https://github.com/facebookresearch/sam3/blob/96914d2425f90a64f45ca977c2b5165418099543/LICENSE) | Frozen scientific backend, separately obtained; source/weights are not rehosted; users obtain them from official sources and comply with their applicable terms. |
| NumPy: 2.3.5, 1.26.4 | Three-clause BSD licence text; [official licence / readme](https://github.com/numpy/numpy/blob/v2.3.5/LICENSE.txt), [official licence / readme](https://raw.githubusercontent.com/numpy/numpy/v1.26.4/LICENSE.txt) | Separate presentation requirements and recorded predictor full-fit environment. |
| Pillow: 12.3.0 | MIT-CMU License; [official licence](https://raw.githubusercontent.com/python-pillow/Pillow/12.3.0/LICENSE) | Presentation dependency |
| Matplotlib: 3.10.8 | Matplotlib License Agreement for versions 1.3.0 and later; the file also retains the earlier agreement; [official licence](https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.8/LICENSE/LICENSE) | Presentation dependency |
| PyTorch: 2.10.0+cu128 | BSD-style three-clause licence with multiple contributor copyright notices; [official licence](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/LICENSE) | Recorded predictor full-fit environment; research dependency obtained separately. |
| CairoSVG: 2.8.2 | GNU Lesser General Public License version 3 text; [official licence](https://raw.githubusercontent.com/Kozea/CairoSVG/2.8.2/LICENSE) | Presentation dependency |
| PyMuPDF: 1.26.7 | AGPL v3; commercial licence alternative described by the same version README; [official licence / readme](https://raw.githubusercontent.com/pymupdf/PyMuPDF/1.26.7/COPYING), [official licence / readme](https://raw.githubusercontent.com/pymupdf/PyMuPDF/1.26.7/README.md) | Presentation/PDF-processing dependency listed in the actual figure requirements; not scientific inference. |

The minimal CPU check requires NumPy; saved-summary rendering adds Matplotlib. The historical predictor setup and earlier full manuscript display environment use further dependencies listed above. The portable release display script does not import PyMuPDF or CairoSVG. Their earlier use is still disclosed rather than silently assigning all dependencies a permissive licence.

SAM is pinned to source commit `96914d2425f90a64f45ca977c2b5165418099543` and the recorded SAM 3.1 multiplex checkpoint identity in `configs/models/sam/`. This revision has its own SAM License, not an assumed Apache licence. Obtain access through the [official versioned repository](https://github.com/facebookresearch/sam3/tree/96914d2425f90a64f45ca977c2b5165418099543) and comply with its current access conditions. No access-control bypass or download is provided here.

PASCAL VOC and COCO remain separate benchmark sources. The [official COCO site](https://cocodataset.org/#download) provides acquisition and terms routes; it does not grant this project blanket rights to redistribute its underlying photographs. PASCAL VOC uses the [official VOC 2012 project page](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/) and its requested project citation: M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn and A. Zisserman, “The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results,” an undated entry, as preserved in the manuscript bibliography. The earlier bounded metadata check could not reach the historical VOC page; no live availability is promised. COCO is cited as T.-Y. Lin et al., “Microsoft COCO: Common Objects in Context,” ECCV (2014), DOI [10.1007/978-3-319-10602-1_48](https://doi.org/10.1007/978-3-319-10602-1_48). Use the official project acquisition routes; original images and labels are not bundled. Dataset archive availability, original picture rights and redistribution clearance were not independently established. No annotation endpoint or original label was opened.

The exact PyTorch +cu128 wheel and CUDA licence bundle, transitive packages (including pycocotools, setuptools and imaging codecs), complete font/TeX notice sets and any commercial PyMuPDF agreement remain unverified. No such components are vendored. An environment recipe does not relicense its runtime under the RAIL-3 MIT grant. Unverified exact binaries, transitive packages, dataset redistribution terms and third-party media are USER_MUST_REVIEW_UPSTREAM_TERMS.

**THIRD_PARTY_BUNDLE_CLEARANCE = NOT_ESTABLISHED.** Before distributing any covered third-party material, preserve its exact applicable licence and notices and resolve the intended distribution with the relevant rights holders. This source-text review is not legal clearance.

The recorded local engineering environment was Python 3.12.3, NumPy 2.5.2, Matplotlib 3.11.1 and PyMuPDF 1.28.2. The version-tag licence checks above concern the bound historical/display requirement versions, not a certification of those newer installed wheels. The candidate includes no wheels; installed/transitive licence clearance remains unverified.
