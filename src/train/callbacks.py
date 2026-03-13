from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class ThroughputCallback(TrainerCallback):
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        _ = (args, state, control, kwargs)
        return None
