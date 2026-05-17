# Reference Notes

This folder intentionally does not include full extracted text from reference papers.

## CholecTrack20 Paper

Used for:

- Dataset motivation and clinical relevance.
- Multi-perspective tracking definitions:
  - visibility
  - intracorporeal
  - intraoperative
- Dataset scale and structure.
- Public detector/tracker benchmark comparison.
- Surgical visual challenge framing such as bleeding, smoke, blur, occlusion, reflection, trocar view, and lens fouling.

Main citation:

Nwoye, C. I., Elgohary, K., Srinivas, A., Zaid, F., Lavanchy, J. L., & Padoy, N. (2025). CholecTrack20: A multi-perspective tracking dataset for surgical tools.

Project page:

https://github.com/CAMMA-public/cholectrack20

## EA-StrongSORT Paper

Used for:

- Detection-based tracking motivation.
- StrongSORT enhancement rationale.
- EfficientNetV2-based ReID feature extractor.
- Efficient Channel Attention in deeper layers.
- GIoU-style association motivation.
- Ablation-study framing for future tracker improvements.

Main citation:

Ghatwary, N., Amer, A., Fayed, S., Magdy, S., Hussein, A., Kadry, R., & Abdelmaksoud, A. I. (n.d.). EA-StrongSORT: An efficient attention StrongSORT framework for detection-based tumor tracking in Cine-MRI TrackRAD2025 dataset.

## Notes For Continuation

The current project paper uses these references for context and methodology discussion. It does not claim that the exact EfficientNetV2/ECA ReID branch from EA-StrongSORT has already been fully implemented; the current implementation uses EfficientDet-D0 detection followed by a StrongSORT/OSNet tracking stage, with the exact EA-StrongSORT ReID replacement listed as future work.
