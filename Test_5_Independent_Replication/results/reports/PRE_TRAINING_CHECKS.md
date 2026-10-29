# Test 5 (independent) - pre-training checks

_Generated 2026-10-11 03:35:19 by 9-10-26_test5_independent.py (sha256 d40fd0340969bcdc...)_

**Leakage gate: PASS** - confirmed cross-split leakage = 0

## Gate checks

| check | result |
|---|---|
| source inventory integrity verified | PASS |
| source-group split isolation verified | PASS |
| original/degraded grouping verified | PASS |
| exact cross-split duplicates investigated (none confirmed) | PASS |
| confirmed cross-split source leakage == 0 | PASS |
| unresolved duplicates documented | PASS |
| no Test 4 / old Test 5 caches or checkpoints loaded | PASS |
| fresh Test 5 cache provenance verified | PASS |
| every dataset has >= 20 real and fake videos in val and test | PASS |

## Source inventory

Originals 10,703; manifest degraded outputs 79,582 (79,582 with valid link metadata); orphan degraded files (not in a manifest, unused) 6,030; exact duplicate files 2007 in 447 sets; hash scope `all`.

| dataset | subset | videos | probe_ok | audio_stream | external_wav | median_width | median_height | median_fps | median_duration_s | degraded_available |
|---|---|---|---|---|---|---|---|---|---|---|
| BioDeepAV | fake | 1693 | 1693 | 1693 | 1693 | 512.000 | 512.000 | 25.000 | 10.200 | 5079.000 |
| BioDeepAV | real | 2010 | 2010 | 2010 | 2010 | 168.000 | 168.000 | 25.000 | 2.720 | 54312.000 |
| FF++ | DeepFakeDetection | 1000 | 1000 | 0 | 0 | 1920.000 | 1080.000 | 24.000 | 30.125 | 3000.000 |
| FF++ | Deepfakes | 1000 | 1000 | 0 | 0 | 854.000 | 480.000 | 25.000 | 16.653 | 2864.000 |
| FF++ | Face2Face | 1000 | 1000 | 0 | 0 | 832.000 | 480.000 | 25.000 | 16.640 | 2864.000 |
| FF++ | FaceShifter | 1000 | 1000 | 0 | 0 | 854.000 | 480.000 | 25.000 | 16.640 | 2864.000 |
| FF++ | FaceSwap | 1000 | 1000 | 0 | 0 | 854.000 | 480.000 | 25.000 | 13.833 | 2864.000 |
| FF++ | NeuralTextures | 1000 | 1000 | 0 | 0 | 832.000 | 480.000 | 25.000 | 13.833 | 2864.000 |
| FF++ | original | 1000 | 1000 | 0 | 0 | 854.000 | 480.000 | 25.000 | 16.640 | 2871.000 |

### Identity metadata

| dataset | subset | identity_status | n |
|---|---|---|---|
| BioDeepAV | fake | unverifiable | 1693 |
| BioDeepAV | real | unverified | 26 |
| BioDeepAV | real | verified | 1984 |
| FF++ | DeepFakeDetection | verified | 1000 |
| FF++ | Deepfakes | verified | 1000 |
| FF++ | Face2Face | verified | 1000 |
| FF++ | FaceShifter | verified | 1000 |
| FF++ | FaceSwap | verified | 1000 |
| FF++ | NeuralTextures | verified | 1000 |
| FF++ | original | verified | 1000 |

- BioDeepAV fake videos carry no identity/source metadata (UUID names): identity overlap with real videos cannot be verified from metadata and is screened by face-crop and audio duplicate checks instead.
- FF++ manipulation folders: degradation type/severity counts are identical across Deepfakes/Face2Face/FaceShifter/FaceSwap/NeuralTextures (same random assignment per sequence id) - documented, not leakage.

## Preprocessing self-tests

Deterministic re-extraction: True; shapes ok: True; label-free signatures: True; clean clips ok/failed: {'ok': 10702, 'failed': 1}.

Input properties that still separate the classes after preprocessing (dataset properties, potential shortcuts):

| dataset | feature | auc_feature_vs_label | mean_real | mean_fake |
|---|---|---|---|---|
| BioDeepAV | native_face_px | 0.993 | 158.419 | 509.876 |
| BioDeepAV | width | 0.990 | 183.241 | 585.517 |
| BioDeepAV | height | 0.992 | 183.686 | 585.517 |

## Grouping and split

Seed 20261009; 787 groups; 18604 content merges before splitting; 1 originals excluded; split hash `6ae9fe14ca0dbd44...`.

| dataset | subset | test | train | val |
|---|---|---|---|---|
| BioDeepAV | fake | 1006.000 | 453.000 | 233.000 |
| BioDeepAV | real | 414.000 | 1396.000 | 200.000 |
| FF++ | DeepFakeDetection | 838.000 | 63.000 | 99.000 |
| FF++ | Deepfakes | 200.000 | 700.000 | 100.000 |
| FF++ | Face2Face | 200.000 | 700.000 | 100.000 |
| FF++ | FaceShifter | 200.000 | 700.000 | 100.000 |
| FF++ | FaceSwap | 200.000 | 700.000 | 100.000 |
| FF++ | NeuralTextures | 200.000 | 700.000 | 100.000 |
| FF++ | original | 200.000 | 700.000 | 100.000 |

Largest groups:

| group_id | n_videos | stratum | merge_reasons | split |
|---|---|---|---|---|
| Gfee9d2cd274d | 910 | BioDeepAV/mixed | {"exact file SHA-256": 158, "exact external .wav SHA-256": 752, "identical decoded face clip": 157, "identical decoded audio waveform": 752, "near-duplicate video (strong)": 3674, "near-duplicate audio (strong)": 6861} | test |
| Gd8024c02978e | 838 | FF++/1 | {"near-duplicate video (strong)": 240} | test |
| G7e32795ce697 | 205 | BioDeepAV/0 | identity tokens only | test |
| Gb16d705f578c | 183 | BioDeepAV/0 | identity tokens only | val |
| Ge7bf609b543c | 178 | BioDeepAV/mixed | {"exact external .wav SHA-256": 136, "identical decoded audio waveform": 138, "near-duplicate video (strong)": 392, "near-duplicate audio (strong)": 601} | val |
| G10d4cb210cdc | 167 | BioDeepAV/0 | identity tokens only | train |
| G9dcc68bc54c2 | 133 | BioDeepAV/0 | {"exact external .wav SHA-256": 3, "identical decoded audio waveform": 6, "near-duplicate audio (strong)": 8} | train |
| G09966a431cfa | 126 | BioDeepAV/0 | {"identical decoded audio waveform": 1, "near-duplicate audio (strong)": 1} | train |
| G0d44f12cfd7b | 122 | BioDeepAV/0 | identity tokens only | train |
| G2ad01c55a6fd | 113 | BioDeepAV/0 | identity tokens only | train |

## Leakage audit

```json
{
 "group_isolation": {
  "groups": 787,
  "groups_in_multiple_splits": 0,
  "identity_tokens": 2793,
  "identity_tokens_in_multiple_splits": 0,
  "examples": []
 },
 "original_degraded_grouping": {
  "degraded_used": 19729,
  "split_mismatch": 0,
  "group_mismatch": 0,
  "used_without_content_link": 0,
  "used_without_metadata_link": 0,
  "sources_with_degraded": 10686
 }
}
```

```json
{
 "exact_file_duplicate_sets_across_splits": 0,
 "exact_file_duplicate_sets_within_split": 344,
 "exact_file_duplicate_sets_across_datasets": 0,
 "decoded_duplicate_sets_across_splits": 0,
 "decoded_duplicate_sets_within_split": 109,
 "decoded_degenerate_duplicates_across_splits_unresolved": 0,
 "content_merges_before_split": 18604,
 "merged_pairs_ending_in_different_splits": 0,
 "near_duplicate_candidates_unmerged": 34983,
 "near_duplicate_candidates_cross_split_unresolved": 2474,
 "near_duplicate_candidates_cross_dataset": 0,
 "near_duplicate_note": "Unmerged candidates have high dHash similarity and NCC >= 0.90 but miss the strong criterion; they are documented for manual investigation (data_audit/near_duplicate_candidates.csv), not counted as confirmed leakage. Fingerprint similarity alone is not proof of a shared source.",
 "method_limits": "Exact screens find byte-identical files and identical decoded clips; the dHash/NCC screen finds re-encoded copies with nearly identical framing; it does NOT detect the same person filmed in different videos."
}
```

```json
{
 "artifact_reads_logged": 40,
 "blocked_forbidden_reads": 0,
 "historical_readonly_reads": 0,
 "cache_files_on_disk": 10,
 "cache_files_without_provenance": [],
 "cache_keys_match_current_data": {
  "faceclean": true,
  "facedeg": true
 },
 "cache_files_linked_or_hardlinked": [],
 "cache_files_same_inode_as_other_runs": [],
 "test4_imported": false,
 "note": "Test 5 never searches other run directories for caches; all caches carry the t5_ namespace, a provenance record and a key derived from this run's inventory hashes."
}
```

## Next step

Training has NOT been started. After reviewing this report run `--phase all`.