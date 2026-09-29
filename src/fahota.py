"""Fragmentation-aware HOTA (FragA, FA-HOTA) as defined in HOTA Sec. 8, plus the PoseTrack21 rule for comparison.

FAHOTA is the faithful implementation; its matching is split into `_matches`
so that FAHOTACompare can score the PoseTrack21 (PT21) fragment rule on exactly the same matches.

Paper (arXiv:2009.07736v2 Sec. 8; the same equations are numbered 33-35 in v1), for a TP c = (gt g, tracker p, t):
    F(c)    = |FrA(c)| / (|TPA(c)| + |FNA(c)| + |FPA(c)|)             (Eq. 33)
    FragA   = 1/|TP| * sum_c F(c)                                     (Eq. 34)
    FA-HOTA = sqrt( sum_c sqrt(A(c) * F(c)) / (|TP| + |FN| + |FP|) )  (Eq. 32)
    "A fragment is a set of TPAs for which there are no FNAs or FPAs between them."
Every timestep strictly between two consecutive TPAs of (g, p) in which g or p is present holds an FNA or FPA of c,
so a fragment breaks on a detection gap (g unmatched), a spurious prediction (p unmatched) or an ID switch.
FAssA = 1/|TP| * sum_c sqrt(A(c) * F(c)), so FA-HOTA = sqrt(DetA * FAssA) combines over sequences by TP weighting.

PT21 (github.com/anDoer/PoseTrack21 @43c9b834b6, eval/posetrack21/posetrack21/trackeval/metrics/hota.py):
    L119 fragmentation_idxs = last_matched_id_a != matched_det_ids   -> new fragment only when g's matched id changes
    L211 FA-HOTA = sqrt(DetA * sqrt(AssA * FragA))                     -> product of averages, not Eq. 32
FAHOTACompare adds FragA_PT21 and FA-HOTA_PT21 (both as PT21 computes them) and FA-HOTA_PT21frag (Eq. 32 with the
PT21 fragments), which splits the FA-HOTA gap into a fragment-rule part and an L211 part.
"""
import numpy as np

for _n, _t in (('float', float), ('int', int), ('bool', bool)):  # trackeval hota.py uses aliases removed in numpy 1.24
    if not hasattr(np, _n):
        setattr(np, _n, _t)

from scipy.optimize import linear_sum_assignment  # noqa: E402
from trackeval import _timing  # noqa: E402
from trackeval.metrics.hota import HOTA  # noqa: E402

EPS = np.finfo('float').eps


def fragment_sizes(new):
    """new[i] marks the first TP of a fragment in a sorted TP list -> size of each TP's fragment."""
    fid = np.cumsum(new) - 1
    return np.bincount(fid)[fid]


class FAHOTA(HOTA):
    """HOTA plus FragA, FAssA and FA-HOTA (HOTA Sec. 8), on HOTA's own matches."""
    FRAG_FIELDS = ['FragA', 'FAssA']  # TP-weighted when combining

    def __init__(self, config=None):
        super().__init__(config)
        self.float_array_fields += FAHOTA.FRAG_FIELDS + ['FA-HOTA']
        self.fields = self.float_array_fields + self.integer_array_fields + self.float_fields
        self.summary_fields = self.float_array_fields + self.float_fields

    @staticmethod
    def _matches(data):
        """HOTA's matching, recomputed (same score matrix, same deterministic solver, so the same matches).
        Returns t, g, p, sim of every Hungarian match (before alpha thresholding), the per-id cumulative presence
        seen[id, k] = number of timesteps < k in which the id is present, and the per-id presence counts."""
        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_seen = np.zeros((data['num_gt_ids'], data['num_timesteps'] + 1))
        tracker_seen = np.zeros((data['num_tracker_ids'], data['num_timesteps'] + 1))
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            similarity = data['similarity_scores'][t]
            sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > 0 + EPS
            sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
            potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou
            gt_seen[gt_ids_t, t + 1] = 1
            tracker_seen[tracker_ids_t, t + 1] = 1
        gt_seen, tracker_seen = gt_seen.cumsum(1), tracker_seen.cumsum(1)
        gt_id_count, tracker_id_count = gt_seen[:, -1:], tracker_seen[np.newaxis, :, -1]
        global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)

        out = [], [], [], []
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            if len(gt_ids_t) == 0 or len(tracker_ids_t) == 0:
                continue
            similarity = data['similarity_scores'][t]
            score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity
            match_rows, match_cols = linear_sum_assignment(-score_mat)
            for lst, v in zip(out, (np.full(len(match_rows), t), gt_ids_t[match_rows], tracker_ids_t[match_cols],
                                    similarity[match_rows, match_cols])):
                lst.append(v)
        mt = tuple(map(np.concatenate, out)) if out[0] else None
        return mt, gt_seen, tracker_seen, gt_id_count, tracker_id_count

    @_timing.time
    def eval_sequence(self, data):
        """Calculates HOTA (via HOTA.eval_sequence) plus the fragmentation fields for one sequence"""
        res = super().eval_sequence(data)  # also zero-initialises the extra fields
        if data['num_tracker_dets'] == 0 or data['num_gt_dets'] == 0:
            return res
        mt, gt_seen, tracker_seen, gt_id_count, tracker_id_count = self._matches(data)
        if mt is None:
            return res
        match_t, match_gt, match_tracker, match_sim = mt
        for a, alpha in enumerate(self.array_labels):
            mask = match_sim >= alpha - EPS  # HOTA's actually_matched_mask
            if not mask.any():
                continue
            # TPs of each (gt id, tracker id) pair, in time order
            order = np.lexsort((match_t[mask], match_tracker[mask], match_gt[mask]))
            t, g, p = match_t[mask][order], match_gt[mask][order], match_tracker[mask][order]
            matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
            np.add.at(matches_count, (g, p), 1)
            tpa_fna_fpa = gt_id_count[g, 0] + tracker_id_count[0, p] - matches_count[g, p]
            # a new fragment starts at the first TPA of a pair, and after any timestep between two consecutive
            # TPAs of the pair in which g or p is present (that timestep holds an FNA or FPA)
            same_pair = (g[1:] == g[:-1]) & (p[1:] == p[:-1])
            g_between = gt_seen[g[:-1], t[1:]] - gt_seen[g[:-1], t[:-1] + 1] > 0  # an FNA of c lies between
            p_between = tracker_seen[p[:-1], t[1:]] - tracker_seen[p[:-1], t[:-1] + 1] > 0  # an FPA lies between
            frag = fragment_sizes(np.r_[True, ~same_pair | g_between | p_between]) / tpa_fna_fpa  # F(c)
            ass = matches_count[g, p] / tpa_fna_fpa  # A(c), as in AssA
            res['FragA'][a] = np.sum(frag) / len(t)
            res['FAssA'][a] = np.sum(np.sqrt(ass * frag)) / len(t)
            self._extra(res, a, t, g, p, ass, tpa_fna_fpa, same_pair, g_between, p_between)
        return self._final(res)

    def _extra(self, res, a, t, g, p, ass, den, same_pair, g_between, p_between):
        """Hook for subclasses: per-alpha extra fields from the (g, p, t)-sorted TPs."""

    def _final(self, res):
        res['FA-HOTA'] = np.sqrt(res['DetA'] * res['FAssA'])
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        return self._combine_fragmentation(all_res, super().combine_sequences(all_res))

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        return self._combine_fragmentation(all_res, super().combine_classes_det_averaged(all_res))

    def _combine_fragmentation(self, all_res, res):
        for field in self.FRAG_FIELDS:
            res[field] = self._combine_weighted_av(all_res, field, res, weight_field='HOTA_TP')
        return self._final(res)


class FAHOTACompare(FAHOTA):
    """FAHOTA plus the PoseTrack21 rule on the same matches (for comparison only)."""
    FRAG_FIELDS = FAHOTA.FRAG_FIELDS + ['FragA_PT21', 'FAssA_PT21frag', 'FragA_FNAonly', 'FragA_FPAonly',
                                        'FragA_floor']

    def __init__(self, config=None):
        super().__init__(config)
        self.float_array_fields += FAHOTACompare.FRAG_FIELDS[2:] + ['FA-HOTA_PT21', 'FA-HOTA_PT21frag']
        self.fields = self.float_array_fields + self.integer_array_fields + self.float_fields
        self.summary_fields = self.float_array_fields + self.float_fields

    def _extra(self, res, a, t, g, p, ass, den, same_pair, g_between, p_between):
        # floor: every fragment of size 1, so F(c) = 1 / den (the least FragA any output with these TPs can have)
        res['FragA_floor'][a] = np.sum(1.0 / den) / len(t)
        # ablations of Eq. 33's break rule: break only on an FNA between (g present), or only on an FPA (p present)
        for f, b in (('FragA_FNAonly', g_between), ('FragA_FPAonly', p_between)):
            res[f][a] = np.sum(fragment_sizes(np.r_[True, ~same_pair | b]) / den) / len(t)
        o = np.lexsort((t, g))  # per gt id in time order; PT21 L119: new fragment when the matched tracker id changes
        go, po = g[o], p[o]
        frag = np.empty(len(t))
        frag[o] = fragment_sizes(np.r_[True, (go[1:] != go[:-1]) | (po[1:] != po[:-1])])
        # PT21 counts fragments per (g, p) but indexes them per g, so a run is a maximal run of g's TPs with one p
        frag /= den
        res['FragA_PT21'][a] = np.sum(frag) / len(t)
        res['FAssA_PT21frag'][a] = np.sum(np.sqrt(ass * frag)) / len(t)

    def _final(self, res):
        res = super()._final(res)
        res['FA-HOTA_PT21'] = np.sqrt(res['DetA'] * np.sqrt(res['AssA'] * res['FragA_PT21']))  # PT21 L211
        res['FA-HOTA_PT21frag'] = np.sqrt(res['DetA'] * res['FAssA_PT21frag'])  # Eq. 32 with PT21 fragments
        return res
