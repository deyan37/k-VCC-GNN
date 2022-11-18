import torch
class Evaluator:
    def eval(self, input_dict):
        correct = (input_dict["y_pred"] == input_dict["y_true"]).sum()
        acc = int(correct) / int(len(input_dict["y_pred"]))
        return acc