import torch
# import os.path as osp
import warnings
# from .base import BaseModel
# from ..smp import splitlen
from PIL import Image
from vlmeval.vlm.smolvlm import SmolVLM


import kornia_vlm_pyo3



def load_image(image_path):
    return image_path


class SmolVLMKornia(SmolVLM):
    def __init__(self, model_path="HuggingFaceTB/SmolVLM-Instruct", **kwargs):
        print("SmolVLM Kornia!")

        # from transformers import AutoProcessor #, Idefics3ForConditionalGeneration

        # assert osp.exists(model_path) or splitlen(model_path) == 2

        # self.processor = AutoProcessor.from_pretrained(model_path)
        # self.model = Idefics3ForConditionalGeneration.from_pretrained(
        #     model_path, torch_dtype=torch.float32, device_map="cuda"
        # )
        kwargs_default = {"max_new_tokens": 2048, "use_cache": True}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config."
        )
        # torch.cuda.empty_cache()

        self.model_rs = kornia_vlm_pyo3.SmolVLMInterface()


    def generate_inner(self, message, dataset=None):
        if dataset in [
            "MMBench_DEV_EN",
            "MMBench_TEST_EN",
            "MMBench_DEV_CN",
            "MMBench_TEST_CN",
            "MMBench",
            "MMBench_CN",
            "MMBench_DEV_EN_V11",
            "MMBench_DEV_CN_V11",
            "MMBench_TEST_EN_V11",
            "MMBench_TEST_CN_V11",
            "MMBench_V11",
            "MMBench_CN_V11",
            "CCBench",
        ]:
            formatted_messages, formatted_images = self.build_prompt_mmbench(message)
        elif dataset in ["MMMU_DEV_VAL", "MMMU_TEST"]:
            formatted_messages, formatted_images = self.build_prompt_mmmu(message)
        elif dataset in ["MathVista_MINI"]:
            formatted_messages, formatted_images = self.build_prompt_mathvista(message)
        elif dataset in ["ChartQA_TEST"]:
            formatted_messages, formatted_images = self.build_prompt_chartqa(message)
        elif dataset in ["DocVQA_VAL", "DocVQA_TEST"]:
            formatted_messages, formatted_images = self.build_prompt_docvqa(message)
        elif dataset in ["TextVQA_VAL", "TextVQA_TEST"]:
            formatted_messages, formatted_images = self.build_prompt_textvqa(message)
        elif dataset in [
            "MME",
            "MMVet",
            "OCRVQA_TEST",
            "OCRVQA_TESTCORE",
            "InfoVQA_VAL",
            "InfoVQA_TEST",
            "OCRBench",
        ]:
            formatted_messages, formatted_images = self.build_prompt_default(
                message, add_brief=True
            )
        elif dataset == "HallusionBench":
            formatted_messages, formatted_images = self.build_prompt_default(
                message, add_yes_or_no=True
            )
        elif dataset in [
            "MMStar",
            "SEEDBench_IMG",
            "AI2D_TEST",
            "ScienceQA_VAL",
            "ScienceQA_TEST",
        ]:
            formatted_messages, formatted_images = self.build_prompt_puremcq(message)
        else:
            formatted_messages, formatted_images = self.build_prompt_default(message)

        # print(formatted_messages)
        # print(formatted_images)
        # return " "

        try:
            self.model_rs.clear_context()

            return self.model_rs.generate_raw(
                formatted_messages,
                self.kwargs["max_new_tokens"],
                formatted_images,
            ).strip()
        except Exception as e:
            print(f"\nError occurred: {e}")
            print("Likely during the sampling process where all the logits are NaN.")
            print(f"formatted_messages: {formatted_messages}, formatted_images: {formatted_images}")

    def build_prompt_default(self, message, add_brief=False, add_yes_or_no=False):
        # from transformers.image_utils import load_image

        prompt, images = "<|im_start|>User:", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += "<image>"
            elif msg["type"] == "text":
                prompt += msg["value"].strip()
        if add_brief:
            prompt += "\nGive a very brief answer."
        if add_yes_or_no:
            prompt += "\nAnswer yes or no."
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def build_prompt_puremcq(self, message):
        # from transformers.image_utils import load_image

        replace_mapping = {
            "\nOptions:": "\nChoices:",
            "Please select the correct answer from the options above.": "Answer with the letter.",
        }

        prompt, images = "<|im_start|>User:", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += "<image>"
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction
        prompt += "<end_of_utterance>\nAssistant: Answer:"
        return prompt, images

    def build_prompt_mt(self, message):
        # from transformers.image_utils import load_image

        prompt, images = "", []
        for msg in message:
            if msg["role"] == "user":
                prompt += "User: "
            elif msg["role"] == "assistant":
                prompt += "Assistant: "
            for item in msg["content"]:
                if item["type"] == "image":
                    img = load_image(item["value"])
                    images.append(img)
                elif item["type"] == "text":
                    prompt += item["value"].strip()
                prompt += "<end_of_utterance>\n"
        return prompt + "Assistant: ", images

    def build_prompt_mmbench(self, message):
        # from transformers.image_utils import load_image

        replace_mapping = {
            "\nOptions:": "\nChoices:",
            "Please select the correct answer from the options above.": "Answer with a letter.",
        }

        prompt, images = "<|im_start|>User:<image>", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                # Swap hint and question
                if instruction.startswith("Hint:"):
                    hint, question = instruction.split("\nQuestion:")
                    question, choices = question.split("\nChoices:")
                    instruction = (
                        "Question:" + question + "\n" + hint + "\nChoices:" + choices
                    )
                prompt += instruction
        prompt += "<end_of_utterance>\nAssistant: Answer:"
        return prompt, images

    def build_prompt_mmmu(self, message):
        # from transformers.image_utils import load_image

        replace_mapping = {
            "Question:": "",
            "Please select the correct answer from the options above.": "Answer with the letter.",
            "\nOptions:": "\nChoices:",
        }

        prompt, images, img_counter = "<|im_start|>User: Question: ", [], 1
        for msg in message:
            if msg["type"] == "image":
                prompt += f"<image {img_counter}>:<image>\n"
                img_counter += 1
        img_counter = 1

        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += f" <image {img_counter}> "
                img_counter += 1
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()
        prompt += "<end_of_utterance>\nAssistant:"
        if "A." in prompt and "B." in prompt:
            prompt += " Answer:"
        return prompt, images

    def build_prompt_mathvista(self, message):
        # from transformers.image_utils import load_image

        replace_mapping = {
            "(A) ": "A. ",
            "(B) ": "B. ",
            "(C) ": "C. ",
            "(D) ": "D. ",
            "(E) ": "E. ",
            "(F) ": "F. ",
            "(G) ": "G. ",
            "(H) ": "H. ",
            "\nOptions:": "\nChoices:",
            "Hint: ": "",
        }

        prompt, images = "<|im_start|>User:<image>", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()

        prompt += "<end_of_utterance>\nAssistant:"
        if "A." in prompt and "B." in prompt:
            prompt += " Answer:"
        return prompt, images

    def build_prompt_chartqa(self, message):
        # from transformers.image_utils import load_image

        prompt = (
            "<|im_start|>User:<image>For the question below, follow the following instructions:\n"
            + "-The answer should contain as few words as possible.\n"
            + "-Don’t paraphrase or reformat the text you see in the image.\n"
            + "-Answer a binary question with Yes or No.\n"
            + "-When asked to give a numerical value, provide a number like 2 instead of Two.\n"
            + "-If the final answer has two or more items, provide it in the list format like [1, 2].\n"
            + "-When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.\n"
            + "-When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17%.\n"
            + "-Don’t include any units in the answer.\n"
            + "-Do not include any full stops at the end of the answer.\n"
            + "-Try to include the full label from the graph when asked about an entity.\n"
            + "Question: "
        )
        images = []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                prompt += msg["value"].strip()
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def build_prompt_docvqa(self, message):
        # from transformers.image_utils import load_image

        prompt = (
            "<|im_start|>User:<image>Give a short and terse answer to the following question. "
            + "Do not paraphrase or reformat the text you see in the image. Do not include any full stops. "
            + "Just give the answer without additional explanation. Question: "
        )

        images = []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                prompt += msg["value"].strip()
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def build_prompt_textvqa(self, message):
        # from transformers.image_utils import load_image

        prompt = (
            "<|im_start|>User:<image>Answer the following question about the image using as few words as possible. "
            + "Follow these additional instructions:\n"
            + "-Always answer a binary question with Yes or No.\n"
            + "-When asked what time it is, reply with the time seen in the image.\n"
            + "-Do not put any full stops at the end of the answer.\n"
            + "-Do not put quotation marks around the answer.\n"
            + "-An answer with one or two words is favorable.\n"
            + "-Do not apply common sense knowledge. The answer can be found in the image.\n"
            + "Question: "
        )
        images = []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                prompt += msg["value"].strip()
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def chat_inner(self, message, dataset=None):
        formatted_messages, formatted_images = self.build_prompt_mt(message)

        # resulting_messages = [
        #     {
        #         "role": "user",
        #         "content": [{"type": "image"}]
        #         + [{"type": "text", "text": formatted_messages}],
        #     }
        # ]
        # prompt = self.processor.apply_chat_template(
        #     resulting_messages, add_generation_prompt=True
        # )

        # print(prompt)
        # print(formatted_images)

        # return " "

        try:
            self.model_rs.clear_context()

            return self.model_rs.generate_raw(
                formatted_messages,
                self.kwargs["max_new_tokens"],
                formatted_images,
            ).strip()
        except Exception as e:
            print(f"Error occurred: {e}")
            print("Likely during the sampling process where all the logits are NaN.")

if __name__ == "__main__":
    pass
    # model = SmolVLMKornia()

    # message_generate = [
    #     {"type": "text", "value": "What is the capital of United States?"},
    #     {"type": "image", "value": "../kornia-rs/.vscode/fuji-mountain-in-autumn.jpg"}
    # ]
    # print(model.generate_inner(message_generate))

    # message_chat = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "value": "What is the capital of France?"},
    #             {"type": "image", "value": "../kornia-rs/.vscode/fuji-mountain-in-autumn.jpg"}
    #         ]
    #     }
    # ]
    # print(model.chat_inner(message_chat))

    # response = kornia_vlm_pyo3.generate_raw(
    #     "<|im_start|>User:<image><image>Can you describe the first image? (it's not a hamburger) What about the second image? <end_of_utterance>\nAssistant:",
    #     400,
    #     [
    #         '/home/ahc/Documents/kornia-rs/.vscode/witness-raw-beauty-majestic-mountain-landscape-meticulously-captured-to-showcase-intricate-details-nature-s-artistry-349750953.jpg',
    #         '/home/ahc/Documents/kornia-rs/.vscode/featured-stovetop-burgers-recipe-300x300.jpg'
    #     ]
    # )
    # print(response)
