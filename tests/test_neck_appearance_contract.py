"""Visible neck appearance is distinct from independently inferred recruitment."""
import unittest
from test_neck_effort_engine import prediction, context, row
from neck_v1_scoring import APPEARANCE_CONSTRUCT, DIMENSIONS, NeckAnalysis, aggregate


def appearance_context(quality=1.):
    ctx = context(quality)
    ctx['versions']['appearance'] = APPEARANCE_CONSTRUCT
    return ctx


class AppearanceContractTests(unittest.TestCase):
    def test_visible_grade_two_survives_unknown_independent_muscle_activity(self):
        output = prediction((1, 0, 2, 0, 0), independent=False)
        item = output.dimensions.anterior_submental_activity
        item.movement_speech_association = False
        result = aggregate(output, appearance_context())
        self.assertEqual(result['dimensions']['anterior_submental_activity']['score'], 2)
        decision = result['diagnostics']['dimensions']['anterior_submental_activity']
        self.assertEqual(decision['scoring_construct'], APPEARANCE_CONSTRUCT)
        self.assertFalse(decision['requires_independent_local_activity'])
        self.assertFalse(decision['model_independent_local_activity'])
        self.assertIsNone(decision['rejection_reason'])
        NeckAnalysis.model_validate(result)
        # Historical effort/recruitment semantics still require independent support.
        self.assertIsNone(aggregate(output, context())['dimensions']['anterior_submental_activity']['score'])

    def test_zero_is_not_replaced_with_a_minimum_grade(self):
        result = aggregate(prediction((0,)*5, independent=False), appearance_context())
        self.assertTrue(all(item['score'] == 0 for item in result['dimensions'].values()))

    def test_visibility_confidence_speech_and_confounds_still_gate_appearance(self):
        for failure in ('visibility', 'confidence', 'speech_context', 'silence',
                        'camera_motion_only', 'normal_articulation_only', 'swallowing_only'):
            with self.subTest(failure=failure):
                output = prediction((0, 0, 2, 0, 0), independent=False)
                ctx = appearance_context()
                item = output.dimensions.anterior_submental_activity
                if failure == 'visibility': ctx['dimension_quality']['anterior_submental_activity'] = .1
                elif failure == 'confidence': item.confidence = .1
                elif failure == 'speech_context': item.speech_context_supported = False
                elif failure == 'silence': ctx['speech_status'] = 'silent'
                else: item.confounds = [failure]
                self.assertIsNone(aggregate(output, ctx)['dimensions']['anterior_submental_activity']['score'])

    def test_neck_appearance_does_not_change_posture_jaw_acceptance(self):
        output = prediction((2, 0, 2, 2, 2), independent=False)
        for name in ('posture_compensation', 'shoulder_recruitment', 'jaw_compensation'):
            getattr(output.dimensions, name).movement_speech_association = False
        result = aggregate(output, appearance_context())
        self.assertEqual(result['dimensions']['anterior_submental_activity']['score'], 2)
        for name in ('posture_compensation', 'shoulder_recruitment', 'jaw_compensation'):
            self.assertIsNone(result['dimensions'][name]['score'])

    def test_jaw_flow_association_is_not_a_visual_appearance_confidence_penalty(self):
        output = prediction((0, 0, 2, 0, 0), independent=False)
        ctx = appearance_context(); ctx['jaw_explained_fraction'] = .99
        result = aggregate(output, ctx)
        self.assertEqual(result['dimensions']['anterior_submental_activity']['confidence'], .95)

    def test_sustained_grade_requires_temporally_distributed_evidence(self):
        output = prediction((0, 0, 2, 0, 0), independent=False)
        ctx = appearance_context()
        ctx['selected_indices'] = [0, 1, 29]
        ctx['frames'] = [dict(index=i, t=i/5) for i in (0, 1, 29)]
        result = aggregate(output, ctx)
        self.assertIsNone(result['dimensions']['anterior_submental_activity']['score'])
        output.dimensions.anterior_submental_activity.frame_indices = [0, 29]
        self.assertEqual(aggregate(output, ctx)['dimensions']['anterior_submental_activity']['score'], 2)

    def test_appearance_does_not_accept_unknown_frame_references(self):
        output = prediction((0, 0, 2, 0, 0), independent=False)
        output.dimensions.anterior_submental_activity.frame_indices = [999]
        with self.assertRaisesRegex(ValueError, 'unsupplied frame'):
            aggregate(output, appearance_context())


if __name__ == '__main__': unittest.main()
