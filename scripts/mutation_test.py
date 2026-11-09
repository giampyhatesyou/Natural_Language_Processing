"""Mutation test of the test suite: every error found by the audit is put back into the code, one at a time, and
the tests must fail. A mutant that survives is an error the tests would not notice again.

    python scripts/mutation_test.py            # all the mutants; exit code 1 if one survives

Kinds of mutants: "audited" re-creates an error of the original implementation (AUDIT.md), "suspected" is the
likely cause of a result whose code was lost, "plausible" is a variant of the same errors in another place.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (kind, description, file, original code, mutated code, tests to run)
MUTANTS = [
    ("audited", "the LSTM baseline is a vanilla RNN", "nlp/lm/models.py",
     "self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)",
     "self.lstm = nn.RNN(emb_size, hidden_size, n_layers, batch_first=True)", "tests/test_lm.py"),
    ("audited", "the dropout model also ties the weights", "nlp/lm/cli.py",
     '"lstm_dropout": {"batch_size_train": 64, "lr": 2.0, "emb_dropout": 0.4, "out_dropout": 0.4},',
     '"lstm_dropout": {"batch_size_train": 64, "lr": 2.0, "emb_dropout": 0.4, "out_dropout": 0.4, '
     '"tie_weights": True},', "tests/test_lm.py"),
    ("audited", "AdamW is applied to the model without dropout", "nlp/lm/cli.py",
     '"weight_decay": 0.01,\n                           "emb_dropout": 0.4, "out_dropout": 0.4},',
     '"weight_decay": 0.01},', "tests/test_lm.py"),
    ("audited", "NT-AvSGD triggers without the non-monotone interval", "nlp/lm/ntavsgd.py",
     "t > self.non_mono and ppl_dev > min(self.logs[:t - self.non_mono])", "t > 0 and ppl_dev > min(self.logs)",
     "tests/test_lm.py"),
    ("audited", "NT-AvSGD selects the SGD weights instead of the averaged ones", "nlp/lm/training.py",
     "        with averaged():\n            stop = stopper.step(", "        if True:\n            stop = stopper.step(",
     "tests/test_lm.py"),
    ("audited", "the intent-slot models are tested on their last epoch", "nlp/slu/training.py",
     "    stopper.restore(model)\n    return stopper, history", "    return stopper, history",
     "tests/test_slu.py tests/test_bert.py"),
    ("audited", "dropout is applied after the intent logits", "nlp/slu/models.py",
     "intent = self.intent_out(self.dropout(last_hidden))", "intent = self.intent_out(last_hidden)",
     "tests/test_slu.py"),
    ("audited", "label ids follow the iteration order of a set", "nlp/slu/data.py",
     "for slot in sorted(slots):", "for slot in slots:", "tests/test_slu.py"),
    ("audited", "the spread over seeds is a population standard deviation", "nlp/common.py",
     "values.std(ddof=1)", "values.std(ddof=0)", "tests/test_slu.py"),
    ("audited", "the checkpoint folder is not created", "nlp/common.py",
     "    path.parent.mkdir(parents=True, exist_ok=True)\n    torch.save(", "    torch.save(", "tests/test_lm.py"),
    ("audited", "every run overwrites the checkpoint of the best one", "nlp/slu/cli.py",
     'if best is None or result["dev_slot_f1"] > best["dev_slot_f1"]:', "if True:", "tests/test_slu.py"),
    ("audited", "BERT labels every sub-token of a word", "nlp/slu/bert_data.py",
     "if word_id is None or word_id == previous:", "if word_id is None:", "tests/test_bert.py"),
    ("suspected", "the initialisation of the LSTM models is applied to BERT", "nlp/slu/cli.py",
     '        if family(config) == "lstm":\n            init_weights(model)',
     "        if True:\n            init_weights(model)", "tests/test_bert.py"),
    ("plausible", "the language models are tested on their last epoch", "nlp/lm/training.py",
     "    stopper.restore(model)\n    return stopper, history", "    return stopper, history", "tests/test_lm.py"),
    ("plausible", "the weights are re-initialised at every epoch", "nlp/slu/training.py",
     "        train_loss = train_epoch(model, train, optimizer, scheduler, clip)",
     "        __import__('nlp.common').common.init_weights(model)\n"
     "        train_loss = train_epoch(model, train, optimizer, scheduler, clip)", "tests/test_bert.py"),
    ("plausible", "early stopping keeps a reference to the weights instead of a copy", "nlp/common.py",
     "self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}",
     "self.best_state = model.state_dict()", "tests/test_lm.py tests/test_slu.py"),
]


def run(kind, description, relative, original, mutated, tests):
    path = ROOT / relative
    source = path.read_text()
    if source.count(original) != 1:
        return f"not applied: the code to mutate is not in {relative}"
    try:
        path.write_text(source.replace(original, mutated))
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *tests.split()],
                                cwd=ROOT, capture_output=True, text=True)
    finally:
        path.write_text(source)
    failed = [line.split(" - ")[0].removeprefix("FAILED ") for line in result.stdout.splitlines()
              if line.startswith("FAILED")]
    return f"killed by {failed[0]}" if result.returncode != 0 and failed else "SURVIVED"


def main():
    survivors = 0
    print("| Kind | Mutant | Result |\n| - | - | - |")
    for kind, description, *mutant in MUTANTS:
        outcome = run(kind, description, *mutant)
        survivors += not outcome.startswith("killed")
        print(f"| {kind} | {description} | {outcome} |", flush=True)
    print(f"\n{len(MUTANTS) - survivors} of {len(MUTANTS)} mutants killed")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
