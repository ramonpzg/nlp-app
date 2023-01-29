import torch
from model import EOIRModel
from data import DataModule


class EOIRPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = EOIRModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.lables = [0, 1]

    def predict(self, text):
        inference_sample = {"text": text}
        processed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    text = "We find no basis for reversing the special inquiry officer's order and remanding the case for a hearing de novo. [MASK] was represented by counsel of his own choosing. He was given a full opportunity to testify in his own behalf."
    predictor = EOIRPredictor("./models/epoch=0-step=267.ckpt")
    print(predictor.predict(text))
