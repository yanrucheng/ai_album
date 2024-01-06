from PIL import Image
from lavis.models import load_model_and_preprocess
import torch

class ImageQuestionAnswerer:
    def __init__(self, model_name, model_type):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name=model_name, model_type=model_type, is_eval=True, device=self.device
        )

    def _get_img(self, img):
        raw_image = img.convert("RGB")
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        return image

    def _get_txt(self, s):
        return self.txt_processors['eval'](s)

    def asks(self, img, questions):
        image = self._get_img(img)
        image_batch = image.repeat(len(questions), 1, 1, 1)
        question_batch = [*map(self._get_txt, questions)]
        return self.model.predict_answers(samples={"image": image_batch, "text_input": question_batch}, inference_method="generate")
        
    def ask(self, img, question):
        image = self._get_img(img)
        q = self._get_txt(question)
        # return self.model.generate({"image": image, "prompt": f"Question: {question} Answer:"})
        return self.model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")

    def rank(self, img, question, candidates):
        image = self._get_img(img)
        question = self._get_txt(question)
        samples = {"image": image, "text_input": question}
        return model.predict_answers(samples, answer_list=candidates, inference_method="rank")

    def caption(self, img, **kw):
        image = self._get_img(img)
        return self.model.generate({"image": image}, **kw)
        
