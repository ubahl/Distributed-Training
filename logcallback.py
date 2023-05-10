# HPML Final Project
# Ryan Friberg & Uma Bahl

from transformers import TrainerCallback
import time

class LogCallBack(TrainerCallback):
    def __init__(self, trainer, run_name):
        self._trainer = trainer
        self.f1    = []
        self.acc   = []
        self.prec  = []
        self.rec   = []
        self.times = []
        self.loss  = []
        self.sps   = []
        self.run_name = run_name

    def on_train_end(self, args, state, control, **kwargs):
        with open(self.run_name, 'w+') as out_file:
            out_file.write("f1=[" + ",".join([str(i) for i in self.f1]) + "]\n")
            out_file.write("accuracy=[" + ",".join([str(i) for i in self.acc]) + "]\n")
            out_file.write("precision=[" + ",".join([str(i) for i in self.prec]) + "]\n")
            out_file.write("recall=[" + ",".join([str(i) for i in self.rec]) + "]\n")
            out_file.write("times=[" + ",".join([str(i) for i in self.times]) + "]\n")
            out_file.write("loss=[" + ",".join([str(i) for i in self.loss]) + "]\n")
            out_file.write("samples/s=[" + ",".join([str(i) for i in self.sps]) + "]\n")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.start_time = time.perf_counter()

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self.f1.append(metrics['eval_f1'])
        self.acc.append(metrics['eval_accuracy'])
        self.prec.append(metrics['eval_precision'])
        self.rec.append(metrics['eval_recall'])
        self.loss.append(metrics['eval_loss'])
        self.sps.append(metrics['eval_samples_per_second'])

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.perf_counter() - self.start_time
        self.times.append(elapsed)
