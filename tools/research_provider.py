"""Trusted local, picklable stage15 provider. No HTTP-controlled file paths."""
from dataclasses import dataclass


@dataclass
class ModelProvider:
    classifier: dict
    localizer: dict
    samples: dict
    stage: int=15

    def __call__(self, sample_id):
        if self.stage==15:
            from tools.stage15_two_component import predict
        elif self.stage==16:
            from tools.stage16_persistence import predict
        else:raise ValueError("unsupported research stage")
        t,z=self.samples[sample_id]
        score,peak,_=predict(t,z,self.classifier,self.localizer)
        return {'clip_logit':score,'candidate_time_sec':peak}


def make_model_provider(model_dir, samples):
    """Admin-only local setup; validates saved model hash, no fitting/extraction."""
    from tools.stage15_two_component import load_bundle
    classifier,localizer=load_bundle(model_dir)
    return ModelProvider(classifier,localizer,samples)
