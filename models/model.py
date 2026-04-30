# Updated models/model.py

# ... previous code ...

class Model:
    def __init__(self, enable_prompt_semantic=True, enable_csm=True, enable_lguaf=True, **kwargs):
        self.enable_prompt_semantic = enable_prompt_semantic
        self.enable_csm = enable_csm
        self.enable_lguaf = enable_lguaf
        # ... other initializations ...

    def some_method(self):
        if self.enable_prompt_semantic:
            self.initialize_prompt_semantic_prior()
        if self.enable_lguaf:
            self.refine_using_lguaf()
        # ... use enable_csm ...
        self.backbone(enable_csm=self.enable_csm)

# ... other code ...