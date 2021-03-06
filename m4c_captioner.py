# Copyright (c) Facebook, Inc. and its affiliates.
# Put this file in the location mmf/models/
from mmf.common.registry import registry
from mmf.models.mma_sr import M4C


@registry.register_model("m4c_captioner")
class M4CCaptioner(M4C):
    def __init__(self, config):
        super().__init__(config)
        print(M4CCaptioner.__bases__)
        self.remove_unk_in_pred = self.config.remove_unk_in_pred
        print("real_m4c_captioner")

    @classmethod
    def config_path(cls):
        return "configs/models/m4c_captioner/defaults.yaml"

    # def _forward_output(self, sample_list, fwd_results):
    #     super()._forward_output(sample_list, fwd_results)
    #
    #     if self.remove_unk_in_pred:
    #          # avoid outputting <unk> in the generated captions
    #          if 'crude_scores' in fwd_results.keys():
    #              fwd_results["crude_scores"][..., self.answer_processor.UNK_IDX] = -1e10
    #          else:
    #             fwd_results["scores"][..., self.answer_processor.UNK_IDX] = -1e10
    #
    #     return fwd_results
